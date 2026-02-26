const form = document.getElementById("uploadForm");
const statusDiv = document.getElementById("status");
const output = document.getElementById("output");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const audioFile = document.getElementById("audio").files[0];
    const task = document.getElementById("task").value;

    if (!audioFile) {
        alert("Please select an audio file");
        return;
    }

    const formData = new FormData();
    formData.append("audio", audioFile);
    formData.append("task", task);

    statusDiv.innerText = "Processing...";
    output.value = "";

    try {
        const response = await fetch("/speech-to-text", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            statusDiv.innerText = "Error";
            output.value = data.error;
        } else {
            statusDiv.innerText = `Done in ${data.time_taken}`;
            output.value = data.text;
        }

    } catch (err) {
        statusDiv.innerText = "Request failed";
        output.value = err.toString();
    }
});