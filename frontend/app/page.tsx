export default function Home() {
  return (
    <main style={{maxWidth: 960, margin: '40px auto', padding: 16, fontFamily: 'system-ui, sans-serif'}}>
      <h1>Kamlan</h1>
      <p>Frontend is live. API endpoints:</p>
      <ul>
        <li><code>/api/healthz</code></li>
        <li><code>/api/hello?name=Kamlan</code></li>
      </ul>
    </main>
  );
}


