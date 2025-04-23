// Get references to the necessary DOM elements
const chatbox = document.getElementById('chatbox');
const messageForm = document.getElementById('message-form');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

// --- Configuration ---
// URL of your backend API endpoint
const API_URL = 'http://192.168.99.152:8000/api/chat'; // Make sure this matches where your backend is running

// --- Helper Functions ---

// Function to format time (e.g., 10:05 AM)
function formatTimestamp(date) {
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

// Basic Markdown Renderer (Bold, Italic, Code Blocks)
function renderSimpleMarkdown(text) {
    // Ensure text is treated as a string
    text = String(text);

    // Code blocks (``` block ```)
    text = text.replace(/```([\s\S]*?)```/g, (match, codeContent) => {
        const escapedCode = codeContent
            .replace(/&/g, "&")
            .replace(/</g, "<")
            .replace(/>/g, ">");
        return `<pre><code>${escapedCode.trim()}</code></pre>`;
    });

    // Bold (**text**)
    text = text.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
    // Italic (*text* or _text_)
    text = text.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.*?)(?<!_)_(?!_)/g, '<i>$1$2</i>'); // More specific italic regex

    // Convert remaining newlines for paragraph breaks (outside code blocks)
    const parts = text.split(/(<pre><code>[\s\S]*?<\/code><\/pre>)/);
    const processedParts = parts.map(part => {
        if (part.startsWith('<pre>')) {
            return part; // Keep code blocks as is
        }
        // Wrap non-empty, non-code block lines in <p> tags
        return part.split('\n').map(line => line.trim() ? `<p>${line.trim()}</p>` : '').join('');
    });

    return processedParts.join('');
}


// Function to add a message to the chatbox
function displayMessage(messageText, sender, isThinking = false, isError = false) {
    const messageWrapper = document.createElement('div');
    messageWrapper.classList.add('message', `${sender}-message`, 'message-enter');

    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');

    if (isThinking) {
        messageWrapper.id = 'thinking-indicator';
        messageWrapper.classList.add('thinking');
        messageContent.innerHTML = `<div class="thinking-dots"><span></span><span></span><span></span></div>`;
    } else if (isError) {
        messageWrapper.classList.add('error-message');
        const paragraph = document.createElement('p');
        paragraph.textContent = messageText; // Keep error text plain
        messageContent.appendChild(paragraph);
    } else {
        // Use innerHTML to render the basic markdown
        messageContent.innerHTML = renderSimpleMarkdown(messageText);
    }

    const messageMeta = document.createElement('div');
    messageMeta.classList.add('message-meta');
    const timestampSpan = document.createElement('span');
    timestampSpan.classList.add('timestamp');
    timestampSpan.textContent = formatTimestamp(new Date());
    messageMeta.appendChild(timestampSpan);

    messageWrapper.appendChild(messageContent);
    messageWrapper.appendChild(messageMeta);
    chatbox.appendChild(messageWrapper);

    chatbox.scrollTo({ top: chatbox.scrollHeight, behavior: 'smooth' });

    setTimeout(() => {
        messageWrapper.classList.remove('message-enter');
    }, 300);
}

// Function to remove the "Thinking..." indicator
function removeThinkingIndicator() {
    const thinkingIndicator = document.getElementById('thinking-indicator');
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}

// Function to enable/disable input form
function setInputDisabled(isDisabled) {
    userInput.disabled = isDisabled;
    sendButton.disabled = isDisabled;
}

// --- Main Logic ---

// Function to get response FROM THE BACKEND API
async function getBotResponse(userMessage) {
    setInputDisabled(true);
    displayMessage("", 'bot', true); // Show thinking indicator

    try {
        // --- Make the actual API call ---
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json' // Indicate we expect JSON back
            },
            body: JSON.stringify({ message: userMessage }) // Send message in correct format
        });
        // --- End API call ---

        removeThinkingIndicator(); // Remove thinking indicator once response headers are received

        // Check if the request was successful (status code 2xx)
        if (!response.ok) {
            // Try to get error details from the response body if possible
            let errorDetails = `HTTP error! Status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetails += ` - ${errorData.detail || JSON.stringify(errorData)}`;
            } catch (e) {
                // If response is not JSON or empty
                errorDetails += ` - ${response.statusText}`;
            }
            throw new Error(errorDetails); // Throw an error to be caught below
        }

        // Parse the JSON response from the backend
        const data = await response.json();

        // Extract the reply text (assuming backend sends {"reply": "..."})
        const botText = data.reply;

        if (botText !== undefined) {
             displayMessage(botText, 'bot');
        } else {
            console.error("API response missing 'reply' field:", data);
            displayMessage("Sorry, I received an unexpected response format from the server.", 'bot', false, true);
        }


    } catch (error) {
        console.error("Error fetching bot response:", error);
        removeThinkingIndicator(); // Ensure thinking indicator is removed on error
        // Display a user-friendly error message in the chat
        // Check if it's a network error vs. an API error message
        let displayError = error.message;
        if (error.message.includes("Failed to fetch")) {
             displayError = "Network error: Could not reach the backend server. Is it running?";
        }
        displayMessage(displayError, 'bot', false, true); // Display error in chat
    } finally {
        setInputDisabled(false); // Re-enable input regardless of success/error
        userInput.focus();
    }
}


// Handle form submission (Send button click)
messageForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const userMessage = userInput.value.trim();

    if (userMessage && !userInput.disabled) {
        displayMessage(userMessage, 'user');
        userInput.value = '';
        getBotResponse(userMessage); // Call the function that now uses fetch
    }
});

// Handle Enter key press for sending message
userInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !userInput.disabled) {
        event.preventDefault();
        messageForm.requestSubmit();
    }
});

// --- Initial Setup ---
function initialGreeting() {
     // Optionally clear previous messages if page is reloaded? Or keep history?
     // For now, just add the greeting.
     // chatbox.innerHTML = ''; // Uncomment to clear history on load
     displayMessage("Hello! How can I assist you with HR tasks today?", 'bot');
     userInput.focus();
}

document.addEventListener('DOMContentLoaded', initialGreeting);