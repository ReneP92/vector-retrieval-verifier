document.addEventListener("DOMContentLoaded", () => {
  const queryInput = document.querySelector("#query");
  const form = document.querySelector(".query-form");
  const themeToggle = document.querySelector("[data-theme-toggle]");
  const themeLabel = document.querySelector("[data-theme-label]");

  const setTheme = (darkMode) => {
    document.documentElement.dataset.theme = darkMode ? "dark" : "light";
    themeToggle?.setAttribute("aria-pressed", String(darkMode));
    themeToggle?.setAttribute(
      "aria-label",
      darkMode ? "Disable night mode" : "Enable night mode",
    );
    if (themeLabel) {
      themeLabel.textContent = darkMode ? "Day mode" : "Night mode";
    }
  };

  themeToggle?.addEventListener("click", () => {
    setTheme(document.documentElement.dataset.theme !== "dark");
  });

  document.querySelectorAll("[data-query]").forEach((button) => {
    button.addEventListener("click", () => {
      queryInput.value = button.dataset.query;
      queryInput.focus();
    });
  });

  queryInput?.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      form?.requestSubmit();
    }
  });
});
