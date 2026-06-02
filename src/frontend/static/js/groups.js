const dateFormatter = new Intl.DateTimeFormat("ru-RU", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
});

function toggleDetails(id) {
  const el = document.getElementById(`details-${id}`);
  if (!el) return;
  const shouldLoad = el.classList.contains("d-none");
  el.classList.toggle("d-none");
  if (shouldLoad) {
    loadStudentAttendance(id);
  }
}

async function manualStatus(studentId) {
  const choice = window.prompt("Введите статус отметки ('present' или 'late')", "present");
  if (!choice) return;
  const status = choice.trim().toLowerCase();
  if (status !== "present" && status !== "late") {
    window.alert("Неправильный статус. Используйте 'present' или 'late'.");
    return;
  }

  try {
    const response = await fetch("/manual_status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ student_id: studentId, status }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const err = payload?.error || `Ошибка: HTTP ${response.status}`;
      window.alert(err);
      return;
    }

    const detailsEl = document.getElementById(`details-${studentId}`);
    if (detailsEl && !detailsEl.classList.contains("d-none")) {
      loadStudentAttendance(studentId);
    }

    window.alert("Отметка успешно добавлена.");
  } catch (e) {
    window.alert("Не удалось отправить отметку. Проверьте соединение.");
  }
}

async function loadStudentAttendance(studentId) {
  renderLoadingState(studentId);

  try {
    const response = await fetch(`/students/${studentId}/attendance`);
    if (!response.ok) {
      throw new Error("request_failed");
    }

    const payload = await response.json();
    renderAttendance(studentId, payload);
  } catch (error) {
    renderErrorState(studentId);
  }
}

function renderLoadingState(studentId) {
  function renderSkeletonEntry() {
    return `
      <div class="attendance-metric skeleton">
        <div class="skeleton-text wide"></div>
        <div class="skeleton-text medium"></div>
      </div>
      <div class="attendance-metric skeleton">
        <div class="skeleton-text medium"></div>
        <div class="skeleton-text narrow"></div>
      </div>
      <div class="attendance-metric skeleton">
        <div class="skeleton-text wide"></div>
        <div class="skeleton-text medium"></div>
      </div>
    `;
  }

  setElementHtml(`attendance-summary-${studentId}`, Array(3).fill(renderSkeletonEntry()).join(''));
  setElementHtml(`late-list-${studentId}`, "");
  setElementHtml(`absent-list-${studentId}`, "");
}

function renderErrorState(studentId) {
  setElementHtml(
    `attendance-summary-${studentId}`,
    '<div class="attendance-empty">Не удалось загрузить посещаемость.</div>',
  );
  setElementHtml(`late-list-${studentId}`, "");
  setElementHtml(`absent-list-${studentId}`, "");
}

function renderAttendance(studentId, payload) {
  const summary = payload.summary || {};

  setElementHtml(
    `attendance-summary-${studentId}`,
    `
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.attendance_rate ?? 0}%</span>
        <span class="attendance-metric-label">Посещаемость</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.attended_days ?? 0}</span>
        <span class="attendance-metric-label">Был</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.on_time_days ?? 0}</span>
        <span class="attendance-metric-label">Без опозданий</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.late_days ?? 0}</span>
        <span class="attendance-metric-label">Опоздал</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.absent_days ?? 0}</span>
        <span class="attendance-metric-label">Пропустил</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.lesson_days ?? 0}</span>
        <span class="attendance-metric-label">Всего дней</span>
      </div>
    `,
  );

  const badge = document.querySelector(
    `.engagement-badge[data-student-id="${studentId}"]`,
  );
  if (badge && payload.engagement && payload.engagement.latest) {
    const level = payload.engagement.latest.engagement_score || "unknown";
    const levelMap = {
      high: "Высокая",
      medium: "Средняя",
      low: "Низкая",
      unknown: "Неизвестно",
    };
    badge.textContent = levelMap[level] || "Неизвестно";
    badge.className = `engagement-badge ${level}`;
  }

  if (payload.engagement && Array.isArray(payload.engagement.history)) {
    renderEngagementChart(studentId, payload.engagement.history);
  } else {
    destroyEngagementChart(studentId);
  }

  renderLateList(studentId, payload.late_arrivals || []);
  renderAbsenceList(studentId, payload.absences || []);
  // Initialize counters and tilt effects for freshly rendered content
  initAnimatedCounters(studentId);
  initVanillaTilt();
}

// VanillaTilt initialization (optional, graceful fallback)
function initVanillaTilt() {
  try {
    if (!window.VanillaTilt) return;
    VanillaTilt.init(document.querySelectorAll('.student-card, .surface-panel, .sidebar-card'), {
      max: 4,
      speed: 420,
      glare: true,
      'max-glare': 0.06,
      perspective: 1000,
      scale: 1.01,
    });
  } catch (e) {}
}

function animateCounter(el, target, duration = 1200) {
  const startTime = performance.now();
  const startVal = 0;

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(startVal + (target - startVal) * eased);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function initAnimatedCounters(studentId) {
  const container = document.getElementById(`attendance-summary-${studentId}`);
  if (!container) return;

  const metricEls = container.querySelectorAll('.attendance-metric-value');
  metricEls.forEach(el => {
    const raw = el.textContent.trim();
    // extract number (handles percentages)
    const num = parseInt(raw.replace(/[^0-9\-]/g, ''), 10) || 0;
    el.dataset.target = String(num);
    el.textContent = '0';
  });

  const metricObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.dataset.target, 10) || 0;
        animateCounter(el, target);
        metricObserver.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  metricEls.forEach(el => metricObserver.observe(el));
}

window._engagementCharts = window._engagementCharts || {};

function renderEngagementChart(studentId, history) {
  if (!window.Chart) return;
  const canvasId = `engagement-chart-${studentId}`;
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const mapLevel = (lvl) => {
    if (!lvl) return null;
    const m = { low: 0, medium: 1, high: 2 };
    return typeof lvl === "string" ? m[lvl] ?? null : null;
  };

  const labels = [];
  const data = [];
  history
    .slice()
    .reverse()
    .forEach((rec) => {
      try {
        const dt = new Date(rec.timestamp);
        labels.push(dt.toLocaleString());
        data.push(mapLevel(rec.engagement_score));
      } catch (e) {
        labels.push(rec.timestamp || "");
        data.push(mapLevel(rec.engagement_score));
      }
    });

  const existing = window._engagementCharts[studentId];
  if (existing) {
    existing.data.labels = labels;
    existing.data.datasets[0].data = data;
    existing.update();
    return;
  }

  const chart = new Chart(ctx.getContext("2d"), {
    type: "line",
    data: { labels: labels, datasets: [{ label: "Вовлечённость", data: data, borderColor: "#00b894", backgroundColor: "rgba(0,184,148,0.15)", tension: 0.2, fill: true, pointRadius: 3 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { ticks: { stepSize: 1, callback: function (val) { const map = { 0: "Low", 1: "Medium", 2: "High" }; return map[val] || ""; } }, min: 0, max: 2 },
        x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 } },
      },
      plugins: { legend: { display: false } },
    },
  });

  window._engagementCharts[studentId] = chart;
}

function destroyEngagementChart(studentId) {
  const existing = window._engagementCharts[studentId];
  if (existing) {
    try {
      existing.destroy();
    } catch (e) {}
    delete window._engagementCharts[studentId];
  }
}

function renderLateList(studentId, lateArrivals) {
  if (!lateArrivals.length) {
    setElementHtml(`late-list-${studentId}`, '<div class="attendance-empty">Опозданий пока нет.</div>');
    return;
  }

  setElementHtml(
    `late-list-${studentId}`,
    lateArrivals
      .map(
        (item) => `
          <div class="attendance-row">
            <span>${formatDate(item.date)}</span>
            <span class="badge text-bg-warning">Пришел в ${item.arrived_at}</span>
          </div>
        `,
      )
      .join("")
  );
}

function renderAbsenceList(studentId, absences) {
  if (!absences.length) {
    setElementHtml(`absent-list-${studentId}`, '<div class="attendance-empty">Пропусков пока нет.</div>');
    return;
  }

  setElementHtml(
    `absent-list-${studentId}`,
    absences
      .map(
        (item) => `
          <div class="attendance-row">
            <span>${formatDate(item.date)}</span>
            <span class="badge text-bg-danger">Отсутствовал</span>
          </div>
        `,
      )
      .join("")
  );
}

function setElementHtml(id, html) {
  const el = document.getElementById(id);
  if (el) {
    el.innerHTML = html;
  }
}

function formatDate(value) {
  const [year, month, day] = value.split("-").map(Number);
  return dateFormatter.format(new Date(year, month - 1, day));
}

