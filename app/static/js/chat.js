async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;
    
    addMessage(message, 'user');
    input.value = '';
    
    try {
        const start_time = performance.now();
        const response = await fetch('/generate_full_objective', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({text: message})
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        let result;
        try {
            result = await response.json();
        } catch (jsonError) {
            throw new Error('Failed to parse response as JSON');
        }
        
        const execution_time = (performance.now() - start_time) / 1000;
        const formattedResponse = formatResponse(result);
        addMessage(formattedResponse, 'bot');
    } catch (error) {
        console.error('Error:', error);
        addMessage(`Error: ${error.message}`, 'bot');
    }
}

function formatResponse(result) {
    let formatted = [];
    
    if (result.objective_name) {
        formatted.push(`Objective Name: ${result.objective_name}`);
    }
    
    for (const [key, value] of Object.entries(result)) {
        if (key !== 'objective_name' && key !== 'execution_time_seconds') {
            formatted.push(`${key}: ${value}`);
        }
    }
    
    if (result.execution_time_seconds) {
        formatted.push(`\nExecution Time: ${result.execution_time_seconds.toFixed(2)}s`);
    }
    
    return formatted.join('\n');
}

function addMessage(text, sender) {
    const chatHistory = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.style.whiteSpace = 'pre-wrap';
    messageDiv.textContent = text;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
