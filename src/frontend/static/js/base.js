function initBaseApp() {
    updateClock();
    setInterval(updateClock, 1000);
    setInterval(loadLiveEvents, 2000);
    loadLiveEvents();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBaseApp);
} else {
    initBaseApp();
}

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

async function loadLiveEvents() {
    try {
        const res = await fetch(`/logs`);
        if (!res.ok) return;
        const logs = await res.json();
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
        console.error(e);
    }
}
