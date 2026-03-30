function initRegisterForm() {
  const regForm = document.getElementById("registerForm");
  if (!regForm) return;

  regForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn = regForm.querySelector("button[type='submit']");
    const statusDiv = document.getElementById("regStatus");

    if (btn) {
      btn.disabled = true;
      btn.innerText = "Загрузка...";
    }

    const nameVal = document.getElementById("regName").value.trim();
    const groupVal = document.getElementById("regGroup").value.trim();
    const photosFiles = document.getElementById("regPhoto")?.files;

    if (!nameVal || !groupVal || !photosFiles || photosFiles.length !== 3) {
      alert("Заполните все поля и выберите ровно 3 фото!");
      if (btn) {
        btn.disabled = false;
        btn.innerText = "Добавить ученика";
      }
      return;
    }

    const formData = new FormData();
    formData.append("name", nameVal);
    formData.append("group", groupVal);

    for (let i = 0; i < photosFiles.length; i++) {
      formData.append("photos", photosFiles[i]);
    }

    try {
      const res = await fetch("/register", { method: "POST", body: formData });
      const data = await res.json().catch(() => null);

      if (res.ok && data) {
        document.getElementById("modalBody").innerHTML = `
          <p><strong>Имя:</strong> ${data.name}</p>
          <p><strong>Группа:</strong> ${data.group}</p>
          <p><strong>Фото:</strong><br>
          ${data.photo_paths.map(p => `<img src="${p}" class="img-fluid mb-2" alt="Фото">`).join('')}
          </p>
        `;

        const modal = new bootstrap.Modal(document.getElementById("successModal"));
        modal.show();

        regForm.reset();
        if (statusDiv) statusDiv.innerText = "Студент успешно добавлен";
      } else {
        const errMsg = data?.error || `HTTP ${res.status}`;
        alert("Ошибка сервера: " + errMsg);
        if (statusDiv) statusDiv.innerText = "Ошибка: " + errMsg;
      }
    } catch (err) {
      alert("Ошибка сети");
      if (statusDiv) statusDiv.innerText = "Ошибка сети";
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerText = "Добавить ученика";
      }
    }
  });
}

document.addEventListener("DOMContentLoaded", initRegisterForm);