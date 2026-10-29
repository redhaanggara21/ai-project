document.getElementById("startBtn").addEventListener("click", function() {
    let startBtn = this;
    startBtn.disabled = true;
    
    let resultDiv = document.getElementById("result");
    let progressBar = document.getElementById("progressBar");
    let progressText = document.getElementById("progressText");

    resultDiv.innerHTML = "<p>Training sedang berjalan...</p>";

    // 1. Picu request untuk memulai training di background thread
    fetch("/start-training/")
    .then(response => response.json())
    .then(data => {
        console.log("function start training progress.js");
        console.log("Training response:", data.message);
    })
    .catch(err => {
        console.error("Gagal memulai training:", err);
        resultDiv.innerHTML = "<p style='color: red;'>Gagal memulai proses training.</p>";
        startBtn.disabled = false;
    });

    // 2. Mulai polling secara berkala untuk mengecek progress
    let interval = setInterval(function() {
        fetch("/progress/") 
        .then(response => response.json())
        .then(data => {
            console.log("Progress status:", data); // Untuk memantau status di Console

            let currentEpoch = data.epoch || 0;
            let totalEpochs = data.total_epochs || 5;
            let percent = totalEpochs > 0 ? Math.floor((currentEpoch / totalEpochs) * 100) : 0;
            
            progressBar.style.width = percent + "%";
            progressText.innerText = `Progress: Epoch ${currentEpoch}/${totalEpochs} (${percent}%)`;
            
            // Jika status backend sudah "done" (selesai)
            if (data.status === "done") {
                clearInterval(interval);
                progressBar.style.width = "100%";
                progressText.innerText = "Progress: 100% (Selesai)";

                let metrics = data.metrics || {};
                let acc = data.accuracy || 0;
                let f1 = metrics.f1 || 0;

                // Tampilkan hasil metrik ke modal/resultDiv
                resultDiv.innerHTML = 
                    "<h3>Hasil Training</h3>" +
                    "<p>Accuracy: " + (acc * 100).toFixed(2) + "%</p>" +
                    "<p>Precision: " + (metrics.precision || 0) + "</p>" +
                    "<p>Recall: " + (metrics.recall || 0) + "</p>" +
                    "<p>F1 Score: " + f1.toFixed(2) + "</p>" +
                    "<a href='/download/cnn/' class='btn'>Download CNN Model</a><br>" +
                    "<a href='/download/xgb/' class='btn'>Download XGBoost Model</a>";
                
                startBtn.disabled = false;
            } 
            // Jika backend mengalami error
            else if (data.status === "error") {
                clearInterval(interval);
                let errMessage = (data.metrics && data.metrics.error) ? data.metrics.error : 'Terjadi kesalahan';
                resultDiv.innerHTML = `<p style='color: red;'>Error: ${errMessage}</p>`;
                startBtn.disabled = false;
            }
        })
        .catch(err => {
            console.error("Gagal mengambil progress:", err);
        });
    }, 1000);
});

document.getElementById("stopBtn").onclick = () => fetch("/stop-training/",{method:"POST"});

async function liveProgress(){
    const res = await fetch("/progress/");
    const data = await res.json();
    updateProgressBar(data);
    
    if(data.status === "done"){
        fetch("/get-metrics/").then(r=>r.json()).then(metrics=>{
            // Update scatter 3D
            const scatterData = {datasets:[{label:"Classes", data:metrics.scatter, backgroundColor:['red','green','blue']}]};
            new Chart(document.getElementById('scatter3d-chart').getContext('2d'),{type:'scatter',data:scatterData});
            
            // Update ROC
            metrics.roc.forEach(roc=>{/* plot per class */});
            
            // F1 table
            const tbody = document.querySelector("#f1-table tbody");
            metrics.f1.forEach((score,i)=>{
                const tr = document.createElement("tr");
                tr.innerHTML = `<td>${metrics.classes[i]}</td><td>${score}</td>`;
                tbody.appendChild(tr);
            });
        });
    } else setTimeout(liveProgress,1000);
}

