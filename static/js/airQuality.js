document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();

    // Collect input data
    const data = {
        features: [
            parseFloat(document.getElementById('CO').value),
            parseFloat(document.getElementById('C6H6').value),
            parseFloat(document.getElementById('NMHC').value),
            parseFloat(document.getElementById('NOx_GT').value),
            parseFloat(document.getElementById('NOx').value),
            parseFloat(document.getElementById('NO2_GT').value),
            parseFloat(document.getElementById('NO2').value),
            parseFloat(document.getElementById('O3').value),
            parseFloat(document.getElementById('T').value),
            parseFloat(document.getElementById('AH').value),
        ]
    };

    // Send POST request to Flask backend
    fetch('/air_quality_prediction/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // Display the result
        document.getElementById('result').innerHTML = 'Prediction: ' + data.prediction;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
