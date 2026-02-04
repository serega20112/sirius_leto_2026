const API_BASE = "/api/v1";
const charts = {};

function initApp() {
    updateClock();
    setInterval(updateClock, 1000);
    setInterval(loadLiveEvents, 2000);
    loadLiveEvents();

    const tabEl = document.querySelector('a[href="#v-pills-groups"]');
    if (tabEl) {
        tabEl.addEventListener('click', loadGroups);
    }

    // Инициализация формы регистрации
    initRegisterForm();
}

// Надёжно инициализируем — если документ ещё парсится, подпишемся, иначе вызовем сразу
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    // DOMContentLoaded уже произошёл — вызываем инициализацию сразу
    initApp();
}

/**
 * Обновляет часы в интерфейсе.
 */
function updateClock() {
    const now = new Date();
    const options = {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    const el = document.getElementById("currentTime");
    if (el) el.innerText = now.toLocaleString("ru-RU", options);
}

/**
 * Загружает список живых событий (логи) для боковой панели мониторинга.
 */
async function loadLiveEvents() {
    try {
        console.log("[FRONT] Запрос логов на", `${API_BASE}/monitor/logs`);
        const res = await fetch(`${API_BASE}/monitor/logs`);
        console.log("[FRONT] Ответ /monitor/logs status=", res.status);
        if (!res.ok) {
            const text = await res.text().catch(()=>"(no body)");
            console.warn("[FRONT] /monitor/logs вернул не-ok:", res.status, text);
            return;
        }
        const logs = await res.json();
        console.log("[FRONT] /monitor/logs json:", Array.isArray(logs) ? `items=${logs.length}` : typeof logs, logs);
        const container = document.getElementById("liveEvents");
        if (!container) return;

        container.innerHTML = logs.slice(0, 20).map(l => `
            <div class="p-3 border-bottom border-secondary-subtle">
                <div class="d-flex justify-content-between">
                    <span class="fw-bold text-light">${l.student_name}</span>
                    <span class="small text-secondary">${new Date(l.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="d-flex justify-content-between mt-1">
                    <span class="badge ${l.is_late ? 'bg-danger' : 'bg-success'}">${l.is_late ? 'Опоздал' : 'Вовремя'}</span>
                    <span class="small status-${l.engagement}">${l.engagement.toUpperCase()}</span>
                </div>
            </div>
        `).join("");
    } catch (e) {
        console.error("[FRONT] Ошибка при загрузке логов:", e);
    }
}

/**
 * Загружает список групп и рендерит аккордеон с учениками.
 */
async function loadGroups() {
    try {
        const res = await fetch(`${API_BASE}/monitor/groups`);
        const groups = await res.json();
        const accordion = document.getElementById("groupsAccordion");
        if (!accordion) return;

        accordion.innerHTML = "";

        Object.keys(groups).forEach((groupName, index) => {
            const students = groups[groupName];
            const html = `
                <div class="accordion-item bg-dark border-secondary">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed bg-dark text-white" type="button" data-bs-toggle="collapse" data-bs-target="#group-${index}">
                            Группа ${groupName} <span class="badge bg-secondary ms-2">${students.length}</span>
                        </button>
                    </h2>
                    <div id="group-${index}" class="accordion-collapse collapse" data-bs-parent="#groupsAccordion">
                        <div class="accordion-body">
                            <div class="row g-3">
                                ${students.map(s => renderStudentCard(s)).join("")}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            accordion.insertAdjacentHTML("beforeend", html);
        });
    } catch (e) {
        console.error(e);
    }
}

/**
 * Генерирует HTML карточки ученика.
 */
function renderStudentCard(student) {
    return `
        <div class="col-md-6 col-lg-4">
            <div class="card bg-card border-secondary h-100">
                <div class="card-body">
                    <div class="d-flex align-items-center mb-3">
                        <img src="${student.photo}" alt="Фото ${student.name}" class="rounded-circle me-3 student-avatar">
                        <div>
                            <h6 class="mb-0 text-white">${student.name}</h6>
                            <small class="text-secondary">ID: ${student.id.substring(0, 8)}...</small>
                        </div>
                    </div>
                    <button class="btn btn-sm btn-outline-primary w-100 mb-2" onclick="toggleDetails('${student.id}')">
                        <i class="bi bi-graph-up"></i> Статистика
                    </button>
                    <div id="details-${student.id}" class="d-none mt-3 border-top border-secondary pt-3">
                        <canvas id="chart-${student.id}" class="mb-3" height="100"></canvas>
                        <div class="d-flex gap-2">
                            <button class="btn btn-sm btn-success flex-grow-1" onclick="updateStatus('${student.id}', 'present')">Пришел</button>
                            <button class="btn btn-sm btn-danger flex-grow-1" onclick="updateStatus('${student.id}', 'absent')">Ушел</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Отправляет запрос на изменение статуса в БД.
 */
async function updateStatus(studentId, action) {
    try {
        await fetch(`${API_BASE}/monitor/manual_status`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                student_id: studentId,
                action: action
            })
        });
        alert("Статус обновлен в базе данных");
    } catch (e) {
        alert("Ошибка соединения");
    }
}

/**
 * Раскрывает детали карточки и инициализирует график.
 */
function toggleDetails(id) {
    const el = document.getElementById(`details-${id}`);
    el.classList.toggle("d-none");
    if (!el.classList.contains("d-none")) {
        setTimeout(() => initStudentChart(id), 100);
    }
}

/**
 * Инициализирует график Chart.js для конкретного ученика.
 */
function initStudentChart(id) {
    const ctx = document.getElementById(`chart-${id}`);
    if (!ctx || charts[id]) return;

    charts[id] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['10:00', '10:15', '10:30', '10:45', '11:00'],
            datasets: [{
                label: 'Вовлеченность',
                data: [1, 2, 2, 1, 2],
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    display: false,
                    min: 0,
                    max: 2
                },
                x: {
                    display: false
                }
            }
        }
    });
}

/**
 * Обработка формы регистрации.
 */

function initRegisterForm() {
    const regForm = document.getElementById("registerForm");

    if (!regForm) {
        console.error("Ошибка: Форма регистрации не найдена в HTML!");
        return;
    }

    console.info("Инициализация формы регистрации: найден элемент registerForm");

    // Используем addEventListener для надёжности и возможности нескольких обработчиков
    regForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log("Регистрация: submit обработчик сработал"); // Проверка в консоли браузера

        const btn = e.target.querySelector("button[type='submit']") || e.target.querySelector("button");
        const statusDiv = document.getElementById("regStatus");

        if (btn) {
            btn.disabled = true;
            btn.innerHTML = "Загрузка...";
        }

        const formData = new FormData();
        const nameVal = document.getElementById("regName").value;
        const groupVal = document.getElementById("regGroup").value;
        const photoInput = document.getElementById("regPhoto");
        const photoFile = photoInput && photoInput.files ? photoInput.files[0] : null;

        if (!nameVal || !groupVal || !photoFile) {
            alert("Заполните все поля!");
            if (btn) btn.disabled = false;
            if (btn) btn.innerHTML = "Добавить ученика";
            return;
        }

        formData.append("name", nameVal);
        formData.append("group", groupVal);
        formData.append("photo", photoFile);

        try {
            console.log("Отправка формы /api/v1/auth/register", nameVal, groupVal, photoFile.name);
            const res = await fetch("/api/v1/auth/register", {
                method: "POST",
                body: formData
            });

            let data = null;
            try {
                data = await res.json();
            } catch (parseErr) {
                console.error("Не удалось распарсить JSON ответа", parseErr);
            }

            if (res.ok) {
                alert("Ученик успешно добавлен! ID: " + (data ? data.id : "(нет id в ответе)"));
                e.target.reset();
                if (statusDiv) statusDiv.innerText = "Ученик добавлен";
            } else {
                const errMsg = data && data.error ? data.error : `HTTP ${res.status}`;
                alert("Ошибка сервера: " + errMsg);
                console.error("Ответ сервера:", res.status, data);
                if (statusDiv) statusDiv.innerText = "Ошибка: " + errMsg;
            }
        } catch (err) {
            alert("Ошибка сети (см. консоль)");
            console.error("Network Error:", err);
            if (statusDiv) statusDiv.innerText = "Ошибка сети";
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = "Добавить ученика";
            }
        }
    });
}