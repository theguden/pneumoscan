document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("image");
    const fileName = document.getElementById("fileName");

    if (!dropZone || !fileInput || !fileName) {
        return;
    }

    function updateFileName(file) {
        if (file) {
            fileName.textContent = "Выбран файл: " + file.name;
        } else {
            fileName.textContent = "Файл не выбран";
        }
    }

    fileInput.addEventListener("change", function () {
        updateFileName(fileInput.files[0]);
    });

    dropZone.addEventListener("dragover", function (event) {
        event.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", function (event) {
        event.preventDefault();
        dropZone.classList.remove("drag-over");

        const files = event.dataTransfer.files;

        if (files.length > 0) {
            fileInput.files = files;
            updateFileName(files[0]);
        }
    });
});