function initRegisterForm() {
    const regForm = document.getElementById("registerForm");
    if (!regForm) return;

    regForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = e.target.querySelector("button[type='submit']");
        const statusDiv = document.getElementById("regStatus");
        if (btn) { btn.disabled = true; btn.innerHTML = "Загрузка..."; }

        const nameVal = document.getElementById("regName").value;
        const groupVal = document.getElementById("regGroup").value;
        const photoInput = document.getElementById("regPhoto");
        const photoFile = photoInput?.files?.[0];

        if (!nameVal || !groupVal || !photoFile) {
            alert("Заполните все поля!");
            if (btn) { btn.disabled = false; btn.innerHTML = "Добавить ученика"; }
            return;
        }

        const formData = new FormData();
        formData.append("name", nameVal);
        formData.append("group", groupVal);
        formData.append("photo", photoFile);

        try {
            const res = await fetch("/api/v1/auth/register", { method: "POST", body: formData });
            const data = await res.json().catch(()=>null);

            if (res.ok) {
                alert("Ученик успешно добавлен! ID: " + (data?.id || "(нет id)"));
                e.target.reset();
                if (statusDiv) statusDiv.innerText = "Ученик добавлен";
            } else {
                const errMsg = data?.error || `HTTP ${res.status}`;
                alert("Ошибка сервера: " + errMsg);
                if (statusDiv) statusDiv.innerText = "Ошибка: " + errMsg;
            }
        } catch (err) {
            alert("Ошибка сети");
            if (statusDiv) statusDiv.innerText = "Ошибка сети";
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = "Добавить ученика"; }
        }
    });
}

if (document.getElementById("registerForm")) {
    document.addEventListener('DOMContentLoaded', initRegisterForm);
}
