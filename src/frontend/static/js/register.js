function initRegisterForm() {
  const registerForm = document.getElementById("registerForm");
  const photoInput = document.getElementById("regPhoto");
  if (!registerForm) {
    return;
  }

  if (photoInput) {
    photoInput.addEventListener("change", updatePhotoHint);
    updatePhotoHint();
  }

  registerForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const submitButton = registerForm.querySelector("button[type='submit']");
    const statusNode = document.getElementById("regStatus");

    if (submitButton) {
      submitButton.disabled = true;
      submitButton.innerText = "Сохраняю профиль...";
    }

    const nameValue = document.getElementById("regName")?.value.trim() || "";
    const groupValue = document.getElementById("regGroup")?.value.trim() || "";
    const photoFiles = photoInput?.files;

    if (!nameValue || !groupValue || !photoFiles || photoFiles.length !== 3) {
      const message = "Нужны имя, группа и ровно 3 фотографии.";
      window.alert(message);
      if (statusNode) {
        statusNode.innerText = message;
      }
      resetSubmitButton(submitButton);
      return;
    }

    const formData = new FormData();
    formData.append("name", nameValue);
    formData.append("group", groupValue);

    for (let index = 0; index < photoFiles.length; index += 1) {
      formData.append("photos", photoFiles[index]);
    }

    try {
      const response = await fetch("/register", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => null);

      if (!response.ok || !payload) {
        const errorMessage = payload?.error || `Ошибка сервера: HTTP ${response.status}`;
        if (statusNode) {
          statusNode.innerText = errorMessage;
        }
        window.alert(errorMessage);
        return;
      }

      renderSuccessModal(payload);
      registerForm.reset();
      updatePhotoHint();
      if (statusNode) {
        statusNode.innerText = "Профиль успешно добавлен.";
      }
    } catch (error) {
      if (statusNode) {
        statusNode.innerText = "Не удалось отправить форму. Проверьте соединение.";
      }
      window.alert("Не удалось отправить форму. Проверьте соединение.");
    } finally {
      resetSubmitButton(submitButton);
    }
  });
}

function updatePhotoHint() {
  const hintNode = document.getElementById("regPhotoHint");
  const photoInput = document.getElementById("regPhoto");
  if (!hintNode || !photoInput) {
    return;
  }

  const count = photoInput.files?.length || 0;
  if (count === 0) {
    hintNode.innerText = "Выберите ровно 3 фото с разным углом лица.";
    return;
  }

  if (count === 3) {
    hintNode.innerText = "Файлы готовы к отправке: 3 из 3.";
    return;
  }

  hintNode.innerText = `Сейчас выбрано ${count} из 3 файлов.`;
}

function renderSuccessModal(payload) {
  const modalBody = document.getElementById("modalBody");
  if (!modalBody) {
    return;
  }

  const images = Array.isArray(payload.photo_paths)
    ? payload.photo_paths
        .map(
          (path) => `
            <img src="${escapeHtml(path)}" alt="Фото ученика" />
          `,
        )
        .join("")
    : "";

  modalBody.innerHTML = `
    <p><strong>Имя:</strong> ${escapeHtml(payload.name || "")}</p>
    <p><strong>Группа:</strong> ${escapeHtml(payload.group || "")}</p>
    <div>${images}</div>
  `;

  const modalElement = document.getElementById("successModal");
  if (!modalElement) {
    return;
  }

  const modal = new bootstrap.Modal(modalElement);
  modal.show();
}

function resetSubmitButton(button) {
  if (!button) {
    return;
  }

  button.disabled = false;
  button.innerText = "Добавить ученика";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initRegisterForm);
} else {
  initRegisterForm();
}
