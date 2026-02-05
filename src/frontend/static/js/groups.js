const charts = {};
const API_BASE = "/api/v1";

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
    } catch(e){ console.error(e); }
}

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

function toggleDetails(id) {
    const el = document.getElementById(`details-${id}`);
    if (!el) return;
    el.classList.toggle("d-none");
    if (!el.classList.contains("d-none")) setTimeout(() => initStudentChart(id), 100);
}

function updateStatus(studentId, action) {
    fetch(`${API_BASE}/monitor/manual_status`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({student_id: studentId, action})
    }).then(()=>alert("Статус обновлен"))
      .catch(()=>alert("Ошибка соединения"));
}

function initStudentChart(id) {
    const ctx = document.getElementById(`chart-${id}`);
    if (!ctx || charts[id]) return;

    charts[id] = new Chart(ctx, {
        type:'line',
        data:{labels:['10:00','10:15','10:30','10:45','11:00'],datasets:[{label:'Вовлеченность',data:[1,2,2,1,2],borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.2)',fill:true,tension:0.4}]},
        options:{responsive:true,plugins:{legend:{display:false}},scales:{y:{display:false,min:0,max:2},x:{display:false}}}
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const tabEl = document.querySelector('a[href="/students_and_groups"]');
    if(tabEl) tabEl.addEventListener('click', loadGroups);
});
