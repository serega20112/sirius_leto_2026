function initRegisterForm() {
    const regForm = document.getElementById("registerForm");
    if (!regForm) return;

    regForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = regForm.querySelector("button[type='submit']");
        const statusDiv = document.getElementById("regStatus");

        if (btn) { btn.disabled = true; btn.innerText = "Загрузка..."; }

        const nameVal = document.getElementById("regName").value;
        const groupVal = document.getElementById("regGroup").value;
        const photoFile = document.getElementById("regPhoto")?.files?.[0];

        if (!nameVal || !groupVal || !photoFile) {
            alert("Заполните все поля!");
            if (btn) { btn.disabled = false; btn.innerText = "Добавить ученика"; }
            return;
        }

        const formData = new FormData();
        formData.append("name", nameVal);
        formData.append("group", groupVal);
        formData.append("photo", photoFile);

        try {
            const res = await fetch("/register", { method: "POST", body: formData });
            const data = await res.json().catch(() => null);

            if (res.ok) {
                // заполняем модалку
                document.getElementById("modalBody").innerHTML = `
                    <p><strong>Имя:</strong> ${data.name}</p>
                    <p><strong>Группа:</strong> ${data.group}</p>
                    <p><strong>Фото:</strong><br>
                    <img src="/src/assets/images/${data.photo}" class="img-fluid" alt="Фото">
                `;
                const modal = new bootstrap.Modal(document.getElementById('successModal'));
                modal.show();

                regForm.reset();
                if (statusDiv) statusDiv.innerText = "Ученик успешно добавлен";
            } else {
                const errMsg = data?.error || `HTTP ${res.status}`;
                alert("Ошибка сервера: " + errMsg);
                if (statusDiv) statusDiv.innerText = "Ошибка: " + errMsg;
            }
        } catch (err) {
            alert("Ошибка сети");
            if (statusDiv) statusDiv.innerText = "Ошибка сети";
        } finally {
            if (btn) { btn.disabled = false; btn.innerText = "Добавить ученика"; }
        }
    });
}

document.addEventListener('DOMContentLoaded', initRegisterForm);
