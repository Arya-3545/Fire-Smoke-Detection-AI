setInterval(() => {
    fetch('/status')
    .then(res => res.json())
    .then(data => {
        document.getElementById("status-text").innerText = data.status;
    });
}, 500);