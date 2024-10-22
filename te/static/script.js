document.getElementById('emotionForm').addEventListener('submit', async function (event) {
    event.preventDefault();
    const userInput = document.getElementById('userInput').value;

    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: userInput })
    });

    const result = await response.json();
    document.getElementById('result').textContent = `Emotion: ${result.emotion}`;
    if (response.status === 503) {
        document.getElementById('result').textContent = 'Model is currently loading, please try again later.';
    } else if (!response.ok) {
        document.getElementById('result').textContent = 'An error occurred. Please try again.';
    } else if (result.message) {
        document.getElementById('result').textContent = result.message;
    }
});
