const charts = {};

function toggleDetails(id) {
    const el = document.getElementById(`details-${id}`);
    if (!el) return;
    el.classList.toggle("d-none");
    if (!el.classList.contains("d-none")) setTimeout(() => initStudentChart(id), 100);
}

function updateStatus(studentId, action) {
    fetch("/monitor/manual_status", {
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
        data:{
            labels:['10:00','10:15','10:30','10:45','11:00'],
            datasets:[{
                label:'Вовлеченность',
                data:[1,2,2,1,2],
                borderColor:'#58a6ff',
                backgroundColor:'rgba(88,166,255,0.2)',
                fill:true,
                tension:0.4
            }]
        },
        options:{
            responsive:true,
            plugins:{legend:{display:false}},
            scales:{y:{display:false,min:0,max:2},x:{display:false}}
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    // все группы уже есть, ничего дополнительно не грузим
});
