let predicted = false;

async function predict()
{
    predicted = true;
    const model_name = document.getElementById("model_name").value;
    const start_date = document.getElementById("start_date").value;
    const end_date = document.getElementById("end_date").value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_name: model_name,
            start_date: start_date,
            end_date: end_date
        })
    });
    const image_data = await response.blob();
    const image_url = URL.createObjectURL(image_data);
    document.getElementById("prediction_image").src = image_url;

    document.getElementById("notice").style.display = "block";
}

function no_data() {
    if(!predicted) {
        return;
    }

    const img = document.getElementById("prediction_image");
    img.src = "no_data.jpg";
}