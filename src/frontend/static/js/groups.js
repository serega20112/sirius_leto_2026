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
  setElementHtml(
    `attendance-summary-${studentId}`,
    '<div class="attendance-empty">Загружаю статистику...</div>',
  );
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
        <span class="attendance-metric-label">Присутствовал</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.on_time_days ?? 0}</span>
        <span class="attendance-metric-label">Вовремя</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.late_days ?? 0}</span>
        <span class="attendance-metric-label">Опоздал</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.absent_days ?? 0}</span>
        <span class="attendance-metric-label">Отсутствовал</span>
      </div>
      <div class="attendance-metric">
        <span class="attendance-metric-value">${summary.lesson_days ?? 0}</span>
        <span class="attendance-metric-label">Учебных дней</span>
      </div>
    `,
  );

  renderLateList(studentId, payload.late_arrivals || []);
  renderAbsenceList(studentId, payload.absences || []);
}

function renderLateList(studentId, lateArrivals) {
  if (!lateArrivals.length) {
    setElementHtml(
      `late-list-${studentId}`,
      '<div class="attendance-empty">Опозданий пока нет.</div>',
    );
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
      .join(""),
  );
}

function renderAbsenceList(studentId, absences) {
  if (!absences.length) {
    setElementHtml(
      `absent-list-${studentId}`,
      '<div class="attendance-empty">Пропусков пока нет.</div>',
    );
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
      .join(""),
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
