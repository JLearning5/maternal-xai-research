const RAW_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"];

export class MaternalRiskModel {
  constructor(configUrl = "assets/models/model-config.json") {
    this.configUrl = configUrl;
    this.config = null;
    this.members = [];
  }

  async load() {
    if (!globalThis.ort) {
      throw new Error("The browser inference runtime did not load.");
    }

    const response = await fetch(this.configUrl, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Model configuration could not be loaded (${response.status}).`);
    }

    this.config = await response.json();
    this.members = await Promise.all(
      this.config.members.map(async (member) => {
        const session = await ort.InferenceSession.create(member.url, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        return { ...member, session };
      }),
    );

    return this;
  }

  get ready() {
    return this.members.length > 0;
  }

  get medians() {
    return this.config?.raw_medians ?? {};
  }

  engineer(raw) {
    const age = Number(raw.Age);
    const systolic = Number(raw.SystolicBP);
    const diastolic = Number(raw.DiastolicBP);
    const bs = Number(raw.BS);
    const temperature = Number(raw.BodyTemp);
    const rawHeartRate = Number(raw.HeartRate);
    const heartRate = rawHeartRate >= 30 && rawHeartRate <= 240 ? rawHeartRate : Number.NaN;
    const pulsePressure = systolic - diastolic;
    const meanArterialPressure = (systolic + 2 * diastolic) / 3;

    return [
      age,
      systolic,
      diastolic,
      bs,
      temperature,
      heartRate,
      pulsePressure,
      meanArterialPressure,
      diastolic / systolic,
      heartRate / systolic,
      age * bs,
      meanArterialPressure * bs,
      bs * heartRate,
      systolic >= 140 ? 1 : 0,
      systolic >= 130 ? 1 : 0,
      systolic <= 90 ? 1 : 0,
      bs > 7.8 ? 1 : 0,
      bs > 11 ? 1 : 0,
      temperature > 99.5 ? 1 : 0,
      temperature > 101.3 ? 1 : 0,
      heartRate > 100 ? 1 : 0,
      heartRate < 60 ? 1 : 0,
      age < 20 ? 1 : 0,
      age > 35 ? 1 : 0,
      age < 18 || age > 40 ? 1 : 0,
      Number(systolic >= 140) + Number(bs > 7.8) + Number(temperature > 99.5) + Number(heartRate > 100) + Number(age < 18 || age > 40),
    ];
  }

  prepareForMember(engineered, member) {
    const values = engineered.map((value, index) =>
      Number.isFinite(value) ? value : member.imputer[index],
    );

    if (!member.scaled) return values;
    return values.map((value, index) => (value - member.mean[index]) / member.scale[index]);
  }

  async predict(raw) {
    if (!this.ready) throw new Error("The model is still loading.");
    for (const feature of RAW_FEATURES) {
      if (!Number.isFinite(Number(raw[feature]))) {
        throw new Error(`Enter a valid value for ${feature}.`);
      }
    }

    const engineered = this.engineer(raw);
    const memberProbabilities = [];

    for (const member of this.members) {
      const prepared = this.prepareForMember(engineered, member);
      const input = new ort.Tensor("float32", Float32Array.from(prepared), [1, prepared.length]);
      const feeds = { [member.input]: input };
      const outputs = await member.session.run(feeds);
      const output = outputs[member.output] ?? Object.values(outputs).find((value) => value?.data?.length === 3);
      if (!output?.data || output.data.length < 3) {
        throw new Error(`${member.name} did not return three class probabilities.`);
      }
      const probabilities = Array.from(output.data).slice(-3).map(Number);
      const sum = probabilities.reduce((total, value) => total + value, 0);
      memberProbabilities.push(probabilities.map((value) => (sum > 0 ? value / sum : 1 / 3)));
    }

    const probabilities = [0, 1, 2].map(
      (classIndex) =>
        memberProbabilities.reduce((sum, values) => sum + values[classIndex], 0) /
        memberProbabilities.length,
    );
    const classIndex = probabilities.indexOf(Math.max(...probabilities));

    return {
      label: this.config.class_names[classIndex],
      classIndex,
      confidence: probabilities[classIndex],
      probabilities,
      memberProbabilities,
    };
  }
}

