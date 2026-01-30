const API_BASE = "/api/v1";
let engagementChart;

document.addEventListener('DOMContentLoaded', () => {
    updateClock();
    setInterval(updateClock, 1000);
    loadInitialData();
    initChart();
    setInterval(() => { loadLogs(); }, 3000);
});

function updateClock() {
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' };
    document.getElementById('currentTime').innerText = new Date().toLocaleString('ru-RU', options);
}

async function loadInitialData() {
    await loadStudents();
    await loadLogs();
}

async function loadStudents() {
    try {
        const res = await fetch(`${API_BASE}/monitor/logs`);
        const logs = await res.json();
        const tbody = document.getElementById('studentTableBody');

        const uniqueStudents = {};
        logs.forEach(l => { uniqueStudents[l.student_name] = l; });

        tbody.innerHTML = Object.values(uniqueStudents).map(s => `
            <tr>
                <td class="p-4">
                    <div class="d-flex align-items-center">
                        <div class="avatar-sm bg-primary rounded-circle me-3 d-flex align-items-center justify-content-center" style="width: 40px; height: 40px">
                            ${s.student_name[0]}
                        </div>
                        <span class="fw-bold">${s.student_name}</span>
                    </div>
                </td>
                <td><span class="text-secondary small">Группа</span></td>
                <td>
                    <span class="badge ${s.is_late ? 'bg-danger-subtle text-danger' : 'bg-success-subtle text-success'} px-3 py-2">
                        ${s.is_late ? 'ОПОЗДАЛ' : 'ПРИСУТСТВУЕТ'}
                    </span>
                </td>
                <td class="text-end p-4">
                    <button class="btn btn-outline-secondary btn-sm me-2" onclick="manualUpdate('${s.student_name}', 'absent')">Покинул</button>
                    <button class="btn btn-outline-primary btn-sm" onclick="manualUpdate('${s.student_name}', 'present')">Пришел</button>
                </td>
            </tr>
        `).join('');
    } catch (e) {}
}

async function loadLogs() {
    try {
        const res = await fetch(`${API_BASE}/monitor/logs`);
        const logs = await res.json();
        const container = document.getElementById('liveLogs');

        container.innerHTML = logs.slice(0, 15).map(l => `
            <div class="list-group-item bg-transparent p-4 border-bottom border-secondary-subtle">
                <div class="d-flex justify-content-between mb-1">
                    <span class="fw-bold">${l.student_name}</span>
                    <span class="text-secondary small">${new Date(l.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="d-flex align-items-center justify-content-between mt-2">
                    <span class="small status-${l.engagement}">Вовлеченность: ${l.engagement.toUpperCase()}</span>
                    <span class="badge bg-darker border border-secondary small text-secondary">${l.is_late ? 'Опоздание' : 'Вовремя'}</span>
                </div>
            </div>
        `).join('');

        updateChartData(logs);
    } catch (e) {}
}

document.getElementById('registerForm').onsubmit = async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    btn.disabled = true;

    const formData = new FormData();
    formData.append('name', document.getElementById('regName').value);
    formData.append('group', document.getElementById('regGroup').value);
    formData.append('photo', document.getElementById('regPhoto').files[0]);

    try {
        const res = await fetch(`${API_BASE}/auth/register`, { method: 'POST', body: formData });
        if (res.ok) {
            document.getElementById('regStatus').innerHTML = '<div class="alert alert-success border-0 py-3">Ученик успешно зарегистрирован</div>';
            e.target.reset();
            loadStudents();
        }
    } catch (err) {}
    btn.disabled = false;
};

function initChart() {
    const ctx = document.getElementById('engagementChart').getContext('2d');
    engagementChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Уровень вовлеченности', data: [], borderColor: '#58a6ff', tension: 0.4, fill: true, backgroundColor: '#58a6ff11' }] },
        options: { responsive: true, scales: { y: { min: 0, max: 2, ticks: { callback: v => ['Low', 'Medium', 'High'][v] } } } }
    });
}

function updateChartData(logs) {
    if (!engagementChart) return;
    const map = { 'low': 0, 'medium': 1, 'high': 2 };
    const lastLogs = logs.slice(-10).reverse();
    engagementChart.data.labels = lastLogs.map(l => new Date(l.timestamp).toLocaleTimeString());
    engagementChart.data.datasets[0].data = lastLogs.map(l => map[l.engagement]);
    engagementChart.update('none');
}

function manualUpdate(name, action) {
    alert(`Статус ученика ${name} изменен на: ${action === 'present' ? 'Присутствует' : 'Покинул'}`);
}