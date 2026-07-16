(() => {
  "use strict";

  const nav = document.querySelector(".hybrid-nav");
  if (!nav) return;

  const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
  const sections = links
    .map((link) => document.getElementById(link.hash.slice(1)))
    .filter((section) => section !== null);

  if (sections.length === 0) return;

  /** Marks one contents link as the current document location. */
  const setActive = (sectionId) => {
    links.forEach((link) => {
      const active = link.hash === `#${sectionId}`;
      link.classList.toggle("is-active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };

  /** Finds the last section that has crossed the upper reading marker. */
  const currentSection = () => {
    const marker = window.scrollY + window.innerHeight * 0.28;
    let current = sections[0];
    for (const section of sections) {
      if (section.offsetTop <= marker) current = section;
      else break;
    }
    return current.id;
  };

  let updateScheduled = false;

  /** Coalesces rapid scroll and resize events into one visual update per frame. */
  const scheduleUpdate = () => {
    if (updateScheduled) return;
    updateScheduled = true;
    window.requestAnimationFrame(() => {
      setActive(currentSection());
      updateScheduled = false;
    });
  };

  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate);
  window.addEventListener("hashchange", scheduleUpdate);
  scheduleUpdate();
})();
