const menuButton = document.querySelector("[data-menu-button]");
const siteNav = document.querySelector("[data-site-nav]");

if (menuButton && siteNav) {
  const closeMenu = () => {
    menuButton.setAttribute("aria-expanded", "false");
    siteNav.classList.remove("open");
  };

  menuButton.addEventListener("click", () => {
    const willOpen = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(willOpen));
    siteNav.classList.toggle("open", willOpen);
  });

  siteNav.addEventListener("click", (event) => {
    if (event.target.closest("a")) closeMenu();
  });

  document.addEventListener("click", (event) => {
    if (!siteNav.contains(event.target) && !menuButton.contains(event.target)) closeMenu();
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 980) closeMenu();
  });
}

document.querySelectorAll("[data-year]").forEach((node) => {
  node.textContent = new Date().getFullYear();
});

