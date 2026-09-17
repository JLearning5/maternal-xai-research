import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(root, "node_modules", "onnxruntime-web", "dist");
const target = resolve(root, "dist", "assets", "vendor", "onnxruntime");
const files = [
  "ort.min.js",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];

await mkdir(target, { recursive: true });
await Promise.all(files.map((file) => copyFile(resolve(source, file), resolve(target, file))));
console.log(`Bundled ${files.length} ONNX Runtime Web assets.`);
