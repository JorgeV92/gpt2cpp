const chat = document.getElementById('chat');
const composer = document.getElementById('composer');
const promptBox = document.getElementById('prompt');
const statusLabel = document.getElementById('status');

function addMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  const roleEl = document.createElement('div');
  roleEl.className = 'role';
  roleEl.textContent = role;
  const body = document.createElement('div');
  body.textContent = text;
  wrapper.appendChild(roleEl);
  wrapper.appendChild(body);
  chat.appendChild(wrapper);
  chat.scrollTop = chat.scrollHeight;
  return body;
}

composer.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = promptBox.value.trim();
  if (!text) return;
  addMessage('user', text);
  promptBox.value = '';
  statusLabel.textContent = 'thinking...';
  const body = addMessage('assistant', '');
  try {
    const response = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: text });
    body.textContent = await response.text();
    statusLabel.textContent = 'done';
  } catch (err) {
    body.textContent = `Request failed: ${err}`;
    statusLabel.textContent = 'error';
  }
});
