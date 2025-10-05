(function () {
  const state = {
    frameId: null,
    scrollListenerAttached: false,
    time: 0,
  };

  const applyScrollParallax = () => {
    const scrollOffset = window.scrollY * 0.18;
    document.body.style.backgroundPosition = `center ${scrollOffset}px`;
  };

  const stopAnimation = () => {
    if (state.frameId !== null) {
      cancelAnimationFrame(state.frameId);
      state.frameId = null;
    }
  };

  const disableScrollListener = () => {
    if (state.scrollListenerAttached) {
      window.removeEventListener("scroll", applyScrollParallax);
      state.scrollListenerAttached = false;
    }
  };

  const enableScrollListener = () => {
    if (!state.scrollListenerAttached) {
      window.addEventListener("scroll", applyScrollParallax, { passive: true });
      state.scrollListenerAttached = true;
    }
  };

  const animate = () => {
    state.time += 0.005;
    const driftX = Math.sin(state.time) * 2;
    const driftY = Math.cos(state.time * 0.7) * 4;
    const scrollOffset = window.scrollY * 0.18;
    document.body.style.backgroundPosition = `${50 + driftX}% ${scrollOffset + driftY}px`;
    state.frameId = requestAnimationFrame(animate);
  };

  const applyPreference = (shouldReduce) => {
    stopAnimation();
    disableScrollListener();
    if (shouldReduce) {
      state.time = 0;
      applyScrollParallax();
      enableScrollListener();
    } else {
      animate();
    }
  };

  const cleanup = (prefersReducedMotion, listener) => {
    stopAnimation();
    disableScrollListener();
    if (prefersReducedMotion) {
      if (typeof prefersReducedMotion.removeEventListener === "function") {
        prefersReducedMotion.removeEventListener("change", listener);
      } else if (typeof prefersReducedMotion.removeListener === "function") {
        prefersReducedMotion.removeListener(listener);
      }
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("mobile-menu-toggle");
    const menu = document.getElementById("mobile-menu");

    if (toggle && menu) {
      toggle.addEventListener("click", () => {
        menu.classList.toggle("hidden");
      });
      menu.querySelectorAll("a").forEach((link) => {
        link.addEventListener("click", () => {
          menu.classList.add("hidden");
        });
      });
    }

    const prefersReducedMotion = window.matchMedia
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : null;

    if (!prefersReducedMotion) {
      animate();
      return;
    }

    const motionListener = (event) => {
      const matches = typeof event === "boolean" ? event : event.matches;
      applyPreference(matches);
    };

    applyPreference(prefersReducedMotion.matches);

    if (typeof prefersReducedMotion.addEventListener === "function") {
      prefersReducedMotion.addEventListener("change", motionListener);
    } else if (typeof prefersReducedMotion.addListener === "function") {
      prefersReducedMotion.addListener(motionListener);
    }

    window.addEventListener("beforeunload", () => {
      cleanup(prefersReducedMotion, motionListener);
    });
  });
})();
