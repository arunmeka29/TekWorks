// Vite proxy redirects this to FastAPI (http://127.0.0.1:8000)
const API_URL = '/predict';

export async function predictCluster(payload) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || 'Server error');
  }

  return await response.json(); // { cluster: <int> }
}
