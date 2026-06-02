function initBaseApp() {
  if (document.getElementById("currentTime")) {
    updateClock();
    setInterval(updateClock, 1000);
  }

  if (document.getElementById("attendanceChart")) {
    loadAttendance();
    setInterval(loadAttendance, 2000);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initBaseApp);
} else {
  initBaseApp();
}

function updateClock() {
  const now = new Date();
  const options = {
    weekday: "long",
    day: "2-digit",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  };
  const clock = document.getElementById("currentTime");
  if (clock) {
    clock.innerText = now.toLocaleString("ru-RU", options);
  }
}

async function loadAttendance() {
  try {
    const response = await fetch("/logs");
    if (!response.ok) {
      return;
    }

    const logs = await response.json();
    const container = document.getElementById("attendanceChart");
    if (!container) {
      return;
    }

    logs.sort(
      (first, second) => new Date(second.timestamp) - new Date(first.timestamp),
    );
    updateMetrics(logs);

    if (!logs.length) {
      container.innerHTML = `
        <div class="feed-empty">
          Журнал пока пуст. Как только система зафиксирует ученика, записи появятся здесь.
        </div>
      `;
      return;
    }

    container.innerHTML = logs
      .slice(0, 20)
      .map((log) => renderLogEntry(log))
      .join("");
  } catch (error) {
    const container = document.getElementById("attendanceChart");
    if (container) {
      container.innerHTML = `
        <div class="feed-empty">
          Не удалось получить журнал. Проверьте backend и повторите позже.
        </div>
      `;
    }
  }
}

function renderLogEntry(log) {
  const statusMeta =
    log.status === "late"
      ? { className: "feed-pill-late", label: "Опоздал" }
      : { className: "feed-pill-present", label: "Пришел" };
  const timeLabel = new Date(log.timestamp).toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return `
    <article class="feed-entry">
      <div class="feed-head">
        <div>
          <p class="feed-name">${escapeHtml(log.student_name || "Unknown")}</p>
        </div>
        <span class="feed-time">${timeLabel}</span>
      </div>
      <div class="feed-tags">
        <span class="feed-pill feed-pill-status ${statusMeta.className}">
          ${statusMeta.label}
        </span>
      </div>
    </article>
  `;
}

function updateMetrics(logs) {
  setMetricValue(
    "statPresent",
    logs.filter((log) => log.status !== "late").length,
  );
  setMetricValue(
    "statLate",
    logs.filter((log) => log.status === "late").length,
  );
}

function setMetricValue(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.innerText = String(value);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// UX enhancements: particles, custom cursor, GSAP timeline
(function () {
  function initParticles() {
    try {
      if (!window.THREE) return;
      // Respect reduced motion preference
      if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      const canvas = document.getElementById('bg-canvas');
      if (!canvas) return;
      const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
      camera.position.z = 4;

      const COUNT = 80;
      const positions = new Float32Array(COUNT * 3);
      const base = new Float32Array(COUNT * 3);
      for (let i = 0; i < COUNT; i++) {
        const x = (Math.random() - 0.5) * 10;
        const y = (Math.random() - 0.5) * 6;
        const z = (Math.random() - 0.5) * 4;
        positions[i * 3 + 0] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;
        base[i * 3 + 0] = x;
        base[i * 3 + 1] = y;
        base[i * 3 + 2] = z;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      const material = new THREE.PointsMaterial({ color: 0xA78BFA, size: 0.06, opacity: 0.9, transparent: true, depthWrite: false });
      const points = new THREE.Points(geometry, material);
      scene.add(points);

      function onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }
      window.addEventListener('resize', onResize);

      let start = performance.now();
      function animate() {
        const t = (performance.now() - start) * 0.001;
        const pos = geometry.attributes.position.array;
        for (let i = 0; i < COUNT; i++) {
          pos[i * 3 + 1] = base[i * 3 + 1] + Math.sin(t * 0.6 + i) * 0.04;
          pos[i * 3 + 0] = base[i * 3 + 0] + Math.sin(t * 0.3 + i * 1.7) * 0.02;
        }
        geometry.attributes.position.needsUpdate = true;
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
      }
      animate();
    } catch (e) {
      // fail silently if Three.js isn't available
      console.warn('Particles init failed', e);
    }
  }

  function initCursor() {
    const dot = document.getElementById('cursor-dot');
    const ring = document.getElementById('cursor-ring');
    if (!dot || !ring) return;
    let mx = 0, my = 0, rx = 0, ry = 0;

    document.addEventListener('mousemove', (e) => {
      mx = e.clientX; my = e.clientY;
      dot.style.left = mx + 'px'; dot.style.top = my + 'px';
    });

    function animateRing() {
      rx += (mx - rx) * 0.08; ry += (my - ry) * 0.08;
      ring.style.left = rx + 'px'; ring.style.top = ry + 'px';
      requestAnimationFrame(animateRing);
    }
    animateRing();

    const hoverables = document.querySelectorAll('a, button, .nav-pill, .student-card, .group-trigger');
    hoverables.forEach(el => {
      el.addEventListener('mouseenter', () => document.body.classList.add('cursor-hover'));
      el.addEventListener('mouseleave', () => document.body.classList.remove('cursor-hover'));
    });
  }

  function initGsapPage() {
    try {
      if (!window.gsap) return;
      gsap.registerPlugin(window.ScrollTrigger);
      const tl = gsap.timeline({ defaults: { ease: 'power2.out' } });
      tl.fromTo('.app-sidebar', { x: -20, opacity: 0 }, { x: 0, opacity: 1, duration: 0.9 })
        .fromTo('.nav-pill', { x: -10, opacity: 0 }, { x: 0, opacity: 1, duration: 0.5, stagger: 0.06 }, '-=0.6')
        .fromTo('.eyebrow', { y: 6, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6 }, '-=0.5')
        .fromTo('.page-title', { y: 10, opacity: 0 }, { y: 0, opacity: 1, duration: 0.9 }, '-=0.6')
        .fromTo('.surface-panel, .student-card, .group-item', { y: 12, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, stagger: 0.08 }, '-=0.6');
    } catch (e) {
      // GSAP optional
    }
  }

  function initUxEnhancements() {
    initParticles();
    initCursor();
    initGsapPage();
    // Initialize VanillaTilt globally if available
    try {
      if (window.VanillaTilt) {
        VanillaTilt.init(document.querySelectorAll('.student-card, .surface-panel, .sidebar-card'), {
          max: 4,
          speed: 400,
          glare: true,
          'max-glare': 0.06,
          perspective: 1000,
          scale: 1.01,
        });
      }
    } catch (e) {
      // optional
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initUxEnhancements);
  } else {
    initUxEnhancements();
  }
})();

// HUD clock (updates `hudClock` in the video HUD)
document.addEventListener('DOMContentLoaded', () => {
  const clockEl = document.getElementById('hudClock');
  if (clockEl) {
    function updateClock() {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, '0');
      const m = String(now.getMinutes()).padStart(2, '0');
      const s = String(now.getSeconds()).padStart(2, '0');
      clockEl.textContent = `${h}:${m}:${s}`;
    }
    updateClock();
    setInterval(updateClock, 1000);
  }
});
