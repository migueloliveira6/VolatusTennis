/* VolatusTennis: service worker (app instalável e funcionamento offline).
 *
 * - Estrutura do site (HTML, CSS, JS, fonte, ícones, Chart.js): cache primeiro, com versão.
 * - Dados (*.json de predicts/, analytics/, data/): rede primeiro, para as previsões nunca
 *   virem da cache havendo ligação; a cache só serve de reserva offline.
 * - Navegação: rede primeiro, com o index.html em cache como reserva offline.
 *
 * Ao alterar css/styles.css ou js/app.js: subir o ?v= no index.html e aqui em SHELL,
 * e subir VERSION. A página mostra "Há uma versão nova" quando o service worker novo
 * fica à espera.
 */
const VERSION = 'v2';
const SHELL_CACHE = `volatus-shell-${VERSION}`;
const DATA_CACHE = 'volatus-data';

const SHELL = [
    './',
    'index.html',
    'css/styles.css?v=1',
    'js/app.js?v=2',
    'fonts/Sakana.ttf',
    'favicon.ico',
    'site.webmanifest',
    'img/logo-192.webp',
    'img/favicon-32.png',
    'img/apple-touch-icon.png',
    'img/icon-192.png',
    'img/icon-512.png',
    'img/icon-maskable-192.png',
    'img/icon-maskable-512.png',
    'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js',
];

self.addEventListener('install', event => {
    event.waitUntil(caches.open(SHELL_CACHE).then(cache => cache.addAll(SHELL)));
});

self.addEventListener('activate', event => {
    event.waitUntil((async () => {
        const names = await caches.keys();
        await Promise.all(names
            .filter(name => name.startsWith('volatus-shell-') && name !== SHELL_CACHE)
            .map(name => caches.delete(name)));
        await self.clients.claim();
    })());
});

// A página pede para ativar já a versão nova (botão "Atualizar")
self.addEventListener('message', event => {
    if (event.data === 'SKIP_WAITING') self.skipWaiting();
});

async function networkFirst(request, cacheName, fallbackUrl = null) {
    const cache = await caches.open(cacheName);
    try {
        const response = await fetch(request);
        if (response.ok) cache.put(fallbackUrl || request, response.clone());
        return response;
    } catch (error) {
        const cached = await cache.match(fallbackUrl || request)
            || (fallbackUrl ? await caches.match(fallbackUrl) : null);
        if (cached) return cached;
        throw error;
    }
}

async function cacheFirst(request) {
    const cached = await caches.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok) {
        const cache = await caches.open(SHELL_CACHE);
        cache.put(request, response.clone());
    }
    return response;
}

self.addEventListener('fetch', event => {
    const { request } = event;
    if (request.method !== 'GET') return;
    const url = new URL(request.url);
    const sameOrigin = url.origin === self.location.origin;

    if (request.mode === 'navigate') {
        // guarda sempre a última versão do HTML em 'index.html' para abrir offline
        event.respondWith(networkFirst(request, SHELL_CACHE, 'index.html'));
    } else if (sameOrigin && url.pathname.endsWith('.json')) {
        event.respondWith(networkFirst(request, DATA_CACHE));
    } else if (sameOrigin || url.hostname === 'cdnjs.cloudflare.com') {
        event.respondWith(cacheFirst(request));
    }
    // outros pedidos externos (ex.: Telegram) seguem sem passar pelo service worker
});
