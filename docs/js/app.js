// VolatusTennis: lógica do site (antes inline no index.html)
let allMatches = []; // Armazena todos os jogos
let currentFilter = 'all'; // Filtro atual

// Sinaliza ao loading screen quando os dados essenciais (previsões) estão prontos,
// com sucesso ou erro - em ambos os casos já há algo para mostrar ao utilizador.
let resolvePredictionsReady;
window.predictionsReady = new Promise(resolve => { resolvePredictionsReady = resolve; });

// ==================== CARREGAMENTO DE DADOS ====================
// fetch com verificação do HTTP status (um 404 do GitHub Pages devolve HTML, não JSON)
async function getJSON(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
}

// Caixa de estado por separador: loading, erro (com "Tentar de novo") ou vazio
function setTabStatus(panelId, state, message = '', retry = null) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    let box = panel.querySelector(':scope > .tab-status');
    if (!box) {
        box = document.createElement('div');
        box.className = 'tab-status';
        box.setAttribute('role', 'status');
        box.setAttribute('aria-live', 'polite');
        (panel.querySelector('.stats-header') || panel.firstElementChild).insertAdjacentElement('afterend', box);
    }
    panel.setAttribute('aria-busy', state === 'loading' ? 'true' : 'false');
    box.className = `tab-status is-${state}`;
    box.hidden = state === 'ok';
    box.textContent = '';
    if (state === 'ok') return;
    if (state === 'loading') {
        const spinner = document.createElement('span');
        spinner.className = 'tab-status-spinner';
        spinner.setAttribute('aria-hidden', 'true');
        box.appendChild(spinner);
    }
    const text = document.createElement('span');
    text.textContent = message;
    box.appendChild(text);
    if (retry) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'tab-status-retry';
        button.textContent = 'Tentar de novo';
        button.addEventListener('click', retry);
        box.appendChild(button);
    }
}

// Carregar dados
Promise.all([
    getJSON('predicts/predictions.json'),
    getJSON('predicts/status.json').catch(() => ({ status: 'ready' }))
])
    .then(([data, status]) => {
        allMatches = data;
        renderChips();
        renderStats(data);
        renderPredictions(data, status);
    })
    .catch(error => {
        console.error('Erro ao carregar previsões:', error);
        const container = document.getElementById('predictions');
        container.textContent = '';
        const box = document.createElement('div');
        box.className = 'tab-status is-error';
        box.setAttribute('role', 'alert');
        box.textContent = 'Não foi possível carregar as previsões. ';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'tab-status-retry';
        button.textContent = 'Tentar de novo';
        button.addEventListener('click', () => location.reload());
        box.appendChild(button);
        container.appendChild(box);
    })
    .finally(() => {
        document.getElementById('predictions')?.setAttribute('aria-busy', 'false');
        resolvePredictionsReady();
    });

let onlyValue = false; // chip "Só com valor"

function renderChips() {
    const row = document.getElementById('tournamentChips');
    if (!row) return;
    const tournaments = [...new Set(allMatches.map(m => m.Torneio))].sort();
    const valueCount = allMatches.filter(m => m["Valor Aposta"] > 0).length;
    const chip = (label, active, onClick) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'flt' + (active ? ' on' : '');
        b.setAttribute('aria-pressed', active ? 'true' : 'false');
        b.textContent = label;
        b.addEventListener('click', onClick);
        return b;
    };
    const chips = [chip(`Todos · ${allMatches.length}`, currentFilter === 'all', () => setTournament('all'))];
    if (tournaments.length > 1) {
        tournaments.forEach(t => chips.push(
            chip(`${t} · ${allMatches.filter(m => m.Torneio === t).length}`, currentFilter === t, () => setTournament(t))
        ));
    }
    chips.push(chip(`Só com valor · ${valueCount}`, onlyValue, () => { onlyValue = !onlyValue; filterByTournament(); }));
    row.replaceChildren(...chips);
}

function setTournament(t) {
    currentFilter = t;
    filterByTournament();
}

function filterByTournament() {
    let list = currentFilter === 'all' ? allMatches : allMatches.filter(m => m.Torneio === currentFilter);
    if (onlyValue) list = list.filter(m => m["Valor Aposta"] > 0);
    renderStats(list);
    renderPredictions(list, { status: 'ready' });
    renderChips();
}

function renderStats(data) {
    const totalMatches = data.length;
    const avgConfidence = totalMatches > 0 ? 
        (data.reduce((sum, m) => sum + m["Confiança (%)"], 0) / totalMatches).toFixed(1) : 0;
    const tournaments = [...new Set(data.map(m => m.Torneio))].length;
    const avgRoi = totalMatches > 0 ? 
        (data.reduce((sum, m) => sum + m["ROI Esperado (%)"], 0) / totalMatches).toFixed(1) : 0;

    document.getElementById('totalMatches').textContent = totalMatches;
    document.getElementById('avgConfidence').textContent = avgConfidence + '%';
    document.getElementById('tournaments').textContent = tournaments;
    document.getElementById('avgRoi').textContent = avgRoi + '%';
}

function renderPredictions(data, status = { status: 'ready' }) {
    const container = document.getElementById('predictions');
    container.classList.remove('loading');

    if (status.status === 'no_matches') {
        container.innerHTML = `<p class="mx-empty">${status.message}</p>`;
        return;
    }
    if (data.length === 0) {
        container.innerHTML = '<p class="mx-empty">Nenhum jogo encontrado para este filtro</p>';
        return;
    }

    const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const short = n => String(n || '').trim().split(/\s+/).pop();
    const odd = o => Number.isFinite(Number(o)) && o !== null && o !== '' ? Number(o).toFixed(2) : esc(o);
    const surfaceColor = s => {
        s = String(s || '').toLowerCase();
        return s.includes('clay') ? 'var(--clay)' : s.includes('grass') ? 'var(--grass)' : s.includes('hard') ? 'var(--hard)' : 'var(--muted)';
    };

    container.innerHTML = data.map(match => {
        const p1 = match["Jogador 1"], p2 = match["Jogador 2"];
        const winner = match["Vencedor Previsto"];
        const w1 = winner === p1;
        const conf = Number(match["Confiança (%)"]) || 0;
        const prob1 = w1 ? conf : 100 - conf;
        const prob2 = 100 - prob1;

        const elo = Number(match["ELO Diff"]) || 0; // Jogador 1 - Jogador 2
        const eloText = elo === 0 ? 'Igual' : `${esc(short(elo > 0 ? p1 : p2))} +${Math.round(Math.abs(elo))}`;
        const h2h = String(match.H2H ?? '-');
        const valor = Number(match["Valor Aposta"]) || 0;
        const roi = Number(match["ROI Esperado (%)"]) || 0;

        const oddWin = Number(w1 ? match["Odd 1"] : match["Odd 2"]);
        const implied = oddWin > 0 ? 100 / oddWin : null;
        let note = '';
        if (implied !== null && valor > 0 && implied < conf) {
            note = `O modelo dá ${conf.toFixed(1)}% a ${esc(short(winner))}, mas as odds de ${oddWin.toFixed(2)} implicam apenas ${implied.toFixed(0)}%.`;
        } else if (implied !== null && valor <= 0 && implied > conf) {
            note = `Porquê sem valor? As odds de ${oddWin.toFixed(2)} implicam ${implied.toFixed(0)}% de probabilidade, acima dos ${conf.toFixed(1)}% do modelo.`;
        } else if (implied !== null && valor <= 0) {
            note = `Porquê sem valor? A vantagem do modelo sobre as odds (${(conf - implied).toFixed(1)} p.p.) fica abaixo do mínimo para apostar.`;
        }

        return `
        <article class="mx">
            <div class="mx-top">
                <div class="mx-m"><span class="mx-dot" style="background:${surfaceColor(match.Superfície)}"></span>${esc(match.Torneio)} · ${esc(match.Superfície)}</div>
                <div class="mx-conf"><b>${conf.toFixed(1)}%</b> <span class="mx-m">confiança</span></div>
            </div>

            <div class="mx-players">
                <div class="mx-p ${w1 ? 'win' : ''}">
                    <div class="mx-m tag">${w1 ? 'Previsão do modelo' : '&nbsp;'}</div>
                    <div class="n">${esc(p1)}</div>
                    <div class="mx-m">Odds ${odd(match["Odd 1"])}</div>
                </div>
                <div class="mx-p ${w1 ? '' : 'win'}">
                    <div class="mx-m tag">${w1 ? '&nbsp;' : 'Previsão do modelo'}</div>
                    <div class="n">${esc(p2)}</div>
                    <div class="mx-m">Odds ${odd(match["Odd 2"])}</div>
                </div>
            </div>

            <div class="mx-bar" role="img" aria-label="Probabilidade: ${esc(p1)} ${prob1.toFixed(1)}%, ${esc(p2)} ${prob2.toFixed(1)}%">
                <i class="${w1 ? 'w' : ''}" style="width:${prob1}%"></i>
                <i class="${w1 ? '' : 'w'}" style="width:${prob2}%"></i>
            </div>
            <div class="mx-m mx-bar-l"><span>${esc(short(p1))} ${prob1.toFixed(1)}%</span><span>${esc(short(p2))} ${prob2.toFixed(1)}%</span></div>

            <div class="mx-tiles">
                <div class="mx-t"><span class="mx-m">Diferença ELO</span><b>${eloText}</b></div>
                <div class="mx-t"><span class="mx-m">Confrontos diretos</span><b class="${h2h.length > 14 ? 'sm' : ''}">${esc(h2h)}</b></div>
                <div class="mx-t"><span class="mx-m">Stake sugerido</span><b class="${valor > 0 ? 'pos' : 'zero'}" title="Fração da banca (Kelly fracionado)">${valor > 0 ? (valor * 100).toFixed(1) + '% da banca' : 'Sem valor'}</b></div>
                <div class="mx-t"><span class="mx-m">ROI esperado</span><b class="${roi > 0 ? 'pos' : roi < 0 ? 'neg' : 'zero'}">${roi > 0 ? '+' : ''}${roi.toFixed(1)}%</b></div>
            </div>
            ${note ? `<p class="mx-note">${note}</p>` : ''}
        </article>`;
    }).join('');
}

// ==================== SISTEMA DE TABS ====================
// Padrão WAI-ARIA Tabs: só o separador ativo está na ordem do Tab; setas/Home/End mudam de separador.
function setupTabList(tabs, onSelect) {
    function select(tab, focus = false) {
        tabs.forEach(t => {
            const active = t === tab;
            t.classList.toggle('active', active);
            t.setAttribute('aria-selected', active ? 'true' : 'false');
            t.tabIndex = active ? 0 : -1;
        });
        if (focus) tab.focus();
        onSelect(tab);
    }
    tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => select(tab));
        tab.addEventListener('keydown', event => {
            const keys = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1 };
            if (!(event.key in keys)) return;
            event.preventDefault();
            select(tabs[(keys[event.key] + tabs.length) % tabs.length], true);
        });
    });
}

setupTabList(Array.from(document.querySelectorAll('.nav-tab')), tab => {
    const targetTab = tab.getAttribute('data-tab');
    // o separador fica no URL (#historico, ...): atalhos da app e links diretos
    if (location.hash !== `#${targetTab}`) history.replaceState(null, '', `#${targetTab}`);
    document.querySelectorAll('.tab-content').forEach(panel => {
        const active = panel.id === `tab-${targetTab}`;
        panel.classList.toggle('active', active);
        panel.hidden = !active;
    });

    if (targetTab === 'estatisticas' && !window.statsLoaded) {
        loadStatistics();
    }
    if (targetTab === 'clustering' && !window.clusteringLoaded) {
        loadClustering();
    }
    if (targetTab === 'historico' && !window.trackLoaded) {
        loadTrackRecord();
    }
});

// ==================== CARREGAR ESTATÍSTICAS ====================
let accuracyTrendChart = null;
let confidenceChart = null;
const TREND_GRANULARITY = 'monthly'; // trocar para 'daily' para granularidade diária

function buildTrendSeries(byDate, granularity = 'daily') {
    if (granularity !== 'monthly') {
        return {
            labels: byDate.map(d => d.date),
            data: byDate.map(d => Number(d.accuracy) || 0)
        };
    }

    const monthMap = new Map();
    byDate.forEach(item => {
        const dateKey = String(item.date || '');
        const monthKey = dateKey.slice(0, 7);
        if (!monthKey || monthKey.length !== 7) return;

        const total = Number(item.total_predictions) || 0;
        const correct = Number(item.correct) || 0;
        if (!monthMap.has(monthKey)) {
            monthMap.set(monthKey, { total: 0, correct: 0 });
        }

        const current = monthMap.get(monthKey);
        current.total += total;
        current.correct += correct;
    });

    const labels = Array.from(monthMap.keys()).sort();
    const data = labels.map(month => {
        const item = monthMap.get(month);
        if (!item || item.total <= 0) return 0;
        return Number(((item.correct / item.total) * 100).toFixed(2));
    });

    return { labels, data };
}

async function loadStatistics() {
    if (window.statsLoading) return;
    window.statsLoading = true;
    setTabStatus('tab-estatisticas', 'loading', 'A carregar estatísticas…');
    try {
        // Carregar dados
        const [globalStats, byDate, byConfidence] = await Promise.all([
            getJSON('analytics/global_stats.json'),
            getJSON('analytics/accuracy_by_date.json'),
            getJSON('analytics/accuracy_by_confidence.json')
        ]);

        console.debug('loadStatistics: fetched data', { globalStats, byDateLength: byDate.length, byConfidenceLength: byConfidence.length });

        // Atualizar KPIs
        document.getElementById('kpi-accuracy').textContent = `${globalStats.accuracy}%`;
        document.getElementById('kpi-total-preds').textContent = globalStats.total_predictions;
        document.getElementById('kpi-correct').textContent = globalStats.correct;
        document.getElementById('kpi-wrong').textContent = globalStats.wrong;

        // Atualizar progress bar
        const progressBar = document.getElementById('accuracy-progress-bar');
        const progressLabel = document.getElementById('accuracy-progress-label');
        progressBar.style.width = `${globalStats.accuracy}%`;
        progressLabel.textContent = `${globalStats.accuracy}%`;

        // Se não houver dados temporais, mostrar mensagem e evitar criar gráficos vazios
        if (!byDate || byDate.length === 0) {
            document.querySelector('#tab-estatisticas .charts-section').insertAdjacentHTML('afterbegin',
                `<div style="color: #ffb86b; text-align:center; padding: 20px;">⚠️ Sem dados de evolução por data.</div>`
            );
        } else {
            // Verificar se Chart foi carregado
            if (typeof Chart === 'undefined') throw new Error('Chart.js não carregado (Chart is undefined)');
            const trendSeries = buildTrendSeries(byDate, TREND_GRANULARITY);

            // Gráfico de evolução temporal
            const ctxTrend = document.getElementById('accuracyTrendChart').getContext('2d');
            if (accuracyTrendChart) { accuracyTrendChart.destroy(); }
            accuracyTrendChart = new Chart(ctxTrend, {
                type: 'line',
                data: {
                    labels: trendSeries.labels,
                    datasets: [{
                        label: 'Taxa de Acerto (%)',
                        data: trendSeries.data,
                        borderColor: '#00c853',
                        backgroundColor: 'rgba(0, 200, 83, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 400 },
                    scales: {
                        y: {
                            beginAtZero: true,
                            min: 0,
                            max: 100,
                            ticks: { 
                                color: '#cfcbe6',
                                callback: function(value) { return value + '%'; }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                            ticks: {
                                color: '#cfcbe6',
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#cfcbe6' } },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const y = context.parsed && typeof context.parsed.y === 'number' ? context.parsed.y : (context.raw || 0);
                                    return `${Number(y).toFixed(1)}%`;
                                }
                            }
                        }
                    }
                }
            });
        }

        // Gráfico por faixa de confiança
        if (!byConfidence || byConfidence.length === 0) {
            document.querySelector('#tab-estatisticas .charts-section').insertAdjacentHTML('beforeend',
                `<div style="color: #ffb86b; text-align:center; padding: 20px;">⚠️ Sem dados por faixa de confiança.</div>`
            );
        } else {
            console.debug('render confidence data', byConfidence.map(d => ({ range: d.confidence_range, accuracy: d.accuracy })));
            const ctxConf = document.getElementById('confidenceChart').getContext('2d');
            if (confidenceChart) { confidenceChart.destroy(); }
            confidenceChart = new Chart(ctxConf, {
                type: 'bar',
                data: {
                    labels: byConfidence.map(d => d.confidence_range),
                    datasets: [{
                        label: 'Taxa de Acerto (%)',
                        data: byConfidence.map(d => Number(d.accuracy) || 0),
                        backgroundColor: byConfidence.map((_, i) => ['rgba(164, 102, 234, 0.8)', 'rgba(102, 126, 234, 0.8)', 'rgba(118, 75, 162, 0.8)', 'rgba(103, 58, 183, 0.8)', 'rgba(81, 45, 168, 0.8)'][i % 5]),
                        borderColor: '#a466ea',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 400 },
                    scales: {
                        y: {
                            beginAtZero: true,
                            min: 0,
                            max: 100,
                            ticks: { 
                                color: '#cfcbe6',
                                callback: function(value) { return value + '%'; }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                            ticks: { color: '#cfcbe6' },
                            grid: { display: false }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const y = context.parsed && typeof context.parsed.y === 'number' ? context.parsed.y : (context.raw || 0);
                                    const idx = context.dataIndex || 0;
                                    const bc = byConfidence[idx] || {};
                                    const correct = bc.correct || 0;
                                    const total = bc.total_predictions || 0;
                                    return `${Number(y).toFixed(1)}% (${correct}/${total})`;
                                }
                            }
                        }
                    }
                }
            });
        }

        // Marcar como carregado com sucesso
        window.statsLoaded = true;
        setTabStatus('tab-estatisticas', 'ok');

        // Carregar Top 10 e Tendências de ELO
        await loadTop10AndTrends();

    } catch (error) {
        console.error('Erro ao carregar estatísticas:', error);
        setTabStatus('tab-estatisticas', 'error',
            'Não foi possível carregar as estatísticas. Verifique a ligação e tente de novo.', loadStatistics);
        window.statsLoaded = false;
    } finally {
        window.statsLoading = false;
    }
}

// ==================== TOP 10 E TENDÊNCIAS DE ELO ====================
let eloTrendsChart = null;
let currentSurface = 'hard';
let clusterScatterChart = null;
let clusterDistributionChart = null;
let clusterTrendChart = null;
let clusterRadarChart = null;
let clusteringDashboard = null;
let clusterPlayerNames = [];
let selectedClusterPlayer = 'all';

async function loadTop10AndTrends() {
    try {
        // Carregar dados
        const [top10Data, eloTrendsData] = await Promise.all([
            getJSON('data/top10_by_surface.json'),
            getJSON('data/elo_trends.json')
        ]);

        // Renderizar Top 10
        renderTop10(top10Data, currentSurface);

        // Renderizar Tendências de ELO
        renderEloTrends(eloTrendsData);

        // Abas de superfície (só se ligam uma vez, mesmo que se tente carregar de novo)
        const surfaceTabs = Array.from(document.querySelectorAll('.surface-tab'));
        if (!surfaceTabs[0]?.dataset.bound) {
            surfaceTabs.forEach(tab => { tab.dataset.bound = '1'; });
            setupTabList(surfaceTabs, tab => {
                currentSurface = tab.getAttribute('data-surface');
                renderTop10(top10Data, currentSurface);
            });
        }
    } catch (error) {
        console.error('Erro ao carregar top 10 e tendências:', error);
        const container = document.getElementById('top10-container');
        if (container) {
            container.textContent = '';
            const box = document.createElement('div');
            box.className = 'tab-status is-error';
            box.textContent = 'Não foi possível carregar o Top 10 e a evolução do ELO. ';
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'tab-status-retry';
            button.textContent = 'Tentar de novo';
            button.addEventListener('click', loadTop10AndTrends);
            box.appendChild(button);
            container.appendChild(box);
        }
    }
}

function renderTop10(top10Data, surface) {
    const container = document.getElementById('top10-container');
    if (!container) return;

    const players = top10Data[surface] || [];

    if (players.length === 0) {
        container.innerHTML = '<p style="color: #999;">Sem dados disponíveis para esta superfície</p>';
        return;
    }

    const html = players.map((player, index) => `
        <div class="player-card">
            <div class="player-rank">${index + 1}</div>
            <div class="player-name-card">${player.name}</div>
            <div class="player-elo">${Number(player.elo).toFixed(0)}</div>
            <div class="player-elo-label">ELO Rating</div>
        </div>
    `).join('');

    container.innerHTML = html;
}

function renderEloTrends(eloTrendsData) {
    if (typeof Chart === 'undefined') {
        console.error('Chart.js não carregado');
        return;
    }

    const canvas = document.getElementById('eloTrendsChart');
    if (!canvas) return;

    // Preparar dados para o gráfico
    const datasets = [];
    const colors = ['#667eea', '#c084fc', '#a466ea', '#ff6b9d', '#feca57'];
    let allDates = new Set();

    // Recolher todas as datas
    Object.values(eloTrendsData).forEach(player => {
        if (player.dates) {
            player.dates.forEach(date => allDates.add(date));
        }
    });

    const sortedDates = Array.from(allDates).sort();

    // Criar dataset para cada jogador
    Object.entries(eloTrendsData).forEach(([playerName, data], index) => {
        if (data.dates && data.elos) {
            const eloByDate = {};
            data.dates.forEach((date, i) => {
                eloByDate[date] = data.elos[i];
            });

            const eloData = sortedDates.map(date => eloByDate[date] || null);

            datasets.push({
                label: playerName,
                data: eloData,
                borderColor: colors[index % colors.length],
                backgroundColor: 'transparent',
                tension: 0.4,
                fill: false,
                pointRadius: 3,
                pointHoverRadius: 5,
                borderWidth: 2
            });
        }
    });

    // Destruir gráfico anterior se existir
    if (eloTrendsChart) {
        eloTrendsChart.destroy();
    }

    // Criar novo gráfico
    const ctx = canvas.getContext('2d');
    eloTrendsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sortedDates.map(d => new Date(d).toLocaleDateString('pt-PT')),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            scales: {
                y: {
                    ticks: {
                        color: '#cfcbe6'
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: {
                        color: '#cfcbe6',
                        maxRotation: 45,
                        minRotation: 0
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#cfcbe6' },
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#c084fc',
                    borderWidth: 1
                }
            }
        }
    });
}

async function loadClustering() {
    if (window.clusteringLoading) return;
    window.clusteringLoading = true;
    setTabStatus('tab-clustering', 'loading', 'A carregar os dados de clustering (cerca de 6 MB, pode demorar uns segundos)…');
    try {
        const summary = await getJSON('analytics/clustering_dashboard.json');
        clusteringDashboard = summary;

        populatePlayerFilter(summary.players || []);

        document.getElementById('cluster-total-players').textContent = summary.unique_players ?? '--';
        document.getElementById('cluster-total-seasons').textContent = summary.seasons ?? '--';
        document.getElementById('cluster-total-clusters').textContent = summary.cluster_count ?? '--';
        document.getElementById('cluster-top-style').textContent = summary.top_style || '--';

        document.getElementById('cluster-best-method').textContent = (summary.best_method || '--').toUpperCase();
        const bestParams = summary.best_parameters?.n_clusters ? `${summary.best_parameters.n_clusters} clusters` : '--';
        document.getElementById('cluster-best-method-meta').textContent = bestParams;

        const largestCluster = summary.largest_cluster || {};
        document.getElementById('cluster-largest-size').textContent = largestCluster.size ?? '--';
        document.getElementById('cluster-largest-share').textContent = largestCluster.share_pct != null ? `${largestCluster.share_pct}% do dataset` : '--';

        const bestMetrics = summary.best_metrics || {};
        const silhouette = bestMetrics.silhouette != null ? Number(bestMetrics.silhouette).toFixed(3) : '--';
        document.getElementById('cluster-best-score').textContent = silhouette;
        document.getElementById('cluster-best-score-meta').textContent = bestMetrics.method ? `${bestMetrics.method} | k=${bestMetrics.k}` : '--';

        const clusterCards = (summary.cluster_distribution || []).map(cluster => {
            const positives = (cluster.top_positive_features || []).slice(0, 4);
            const negatives = (cluster.top_negative_features || []).slice(0, 3);
            const representativeMatches = String(cluster.representatives || '').match(/'player_name'\s*:\s*'([^']+)'/g) || [];
            const representatives = representativeMatches
                .map(item => item.replace(/'player_name'\s*:\s*'/, '').replace(/'$/, ''))
                .slice(0, 4)
                .join(', ');

            return `
                <div class="cluster-card">
                    <div class="cluster-card-header">
                        <div class="cluster-badge">C${cluster.cluster}</div>
                        <div class="cluster-style">${cluster.style_label || 'Desconhecido'}</div>
                    </div>
                    <small>${cluster.size} jogadores · ${cluster.share_pct}% do total</small>
                    <div class="chip-list">
                        ${positives.map(feature => `<span class="chip">+ ${feature}</span>`).join('')}
                    </div>
                    ${negatives.length ? `<div class="chip-list">${negatives.map(feature => `<span class="chip">- ${feature}</span>`).join('')}</div>` : ''}
                    ${representatives ? `<div class="cluster-meta-subtitle" style="margin-top:12px;">Exemplos: ${representatives}</div>` : ''}
                </div>
            `;
        }).join('');

        document.getElementById('cluster-cards').innerHTML = clusterCards || '<p style="color: #999;">Sem clusters para mostrar.</p>';

        const styleBreakdown = (summary.style_distribution || []).map(item => `
            <div class="cluster-meta-card">
                <h3>${item.style_label || 'Desconhecido'}</h3>
                <div class="cluster-meta-value">${item.count}</div>
                <div class="cluster-meta-subtitle">${item.share_pct}% dos jogadores</div>
            </div>
        `).join('');
        document.getElementById('style-breakdown').innerHTML = styleBreakdown || '<p style="color: #999;">Sem distribuição de estilos disponível.</p>';

        refreshClusteringCharts();

        window.clusteringLoaded = true;
        setTabStatus('tab-clustering', 'ok');
    } catch (error) {
        console.error('Erro ao carregar clustering:', error);
        setTabStatus('tab-clustering', 'error',
            'Não foi possível carregar os dados de clustering. Verifique a ligação e tente de novo.', loadClustering);
        window.clusteringLoaded = false;
    } finally {
        window.clusteringLoading = false;
    }
}

function populatePlayerFilter(players) {
    const input = document.getElementById('playerSearch');
    const results = document.getElementById('playerSearchResults');
    const clearButton = document.getElementById('clearPlayerSearch');

    if (!input || !results) return;

    clusterPlayerNames = Array.from(
        new Set(
            (players || [])
                .map(player => player.player_name)
                .filter(Boolean)
        )
    ).sort((a, b) => a.localeCompare(b));

    function renderSearchResults(searchTerm = '') {
        const term = searchTerm.trim().toLowerCase();

        const filteredPlayers = term
            ? clusterPlayerNames.filter(name =>
                name.toLowerCase().includes(term)
            )
            : clusterPlayerNames;

        results.innerHTML = '';
        activeOptionIndex = -1;
        input.removeAttribute('aria-activedescendant');

        function addOption(label, value, extraClass = '') {
            const option = document.createElement('div');
            option.className = `player-search-result ${extraClass}`.trim();
            option.id = `player-option-${results.children.length}`;
            option.setAttribute('role', 'option');
            option.setAttribute('aria-selected', 'false');
            option.dataset.value = value;
            option.textContent = label;
            // mousedown em vez de click: escolhe antes de o input perder o foco
            option.addEventListener('mousedown', event => {
                event.preventDefault();
                selectClusterPlayer(value);
            });
            results.appendChild(option);
        }

        addOption('👥 Todos os jogadores', 'all', 'all-players');

        if (filteredPlayers.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'player-search-no-results';
            noResults.setAttribute('role', 'status');
            noResults.textContent = 'Nenhum jogador encontrado.';
            results.appendChild(noResults);
        } else {
            // Limitar resultados para não criar uma lista gigante
            filteredPlayers.slice(0, 50).forEach(name => addOption(name, name));
        }

        results.classList.add('active');
        input.setAttribute('aria-expanded', 'true');
    }

    let activeOptionIndex = -1;

    function closeResults() {
        results.classList.remove('active');
        input.setAttribute('aria-expanded', 'false');
        input.removeAttribute('aria-activedescendant');
        activeOptionIndex = -1;
    }

    function moveActiveOption(step) {
        const options = Array.from(results.querySelectorAll('[role="option"]'));
        if (!options.length) return;
        activeOptionIndex = (activeOptionIndex + step + options.length) % options.length;
        options.forEach((option, i) => {
            const active = i === activeOptionIndex;
            option.classList.toggle('is-active', active);
            option.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        const activeOption = options[activeOptionIndex];
        input.setAttribute('aria-activedescendant', activeOption.id);
        activeOption.scrollIntoView({ block: 'nearest' });
    }

    input.addEventListener('keydown', event => {
        const isOpen = results.classList.contains('active');
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            if (!isOpen) renderSearchResults(input.value);
            moveActiveOption(event.key === 'ArrowDown' ? 1 : -1);
        } else if (event.key === 'Enter' && isOpen) {
            const options = results.querySelectorAll('[role="option"]');
            const target = options[activeOptionIndex] || (input.value.trim() ? options[1] : null);
            if (target) {
                event.preventDefault();
                selectClusterPlayer(target.dataset.value);
            }
        } else if (event.key === 'Escape' && isOpen) {
            event.preventDefault();
            closeResults();
        }
    });

    function selectClusterPlayer(playerName) {
        selectedClusterPlayer = playerName;

        if (playerName === 'all') {
            input.value = '';
            clearButton.style.display = 'none';
        } else {
            input.value = playerName;
            clearButton.style.display = 'block';
        }

        closeResults();

        refreshClusteringCharts();
    }

    input.addEventListener('input', () => {
        clearButton.style.display =
            input.value.trim() ? 'block' : 'none';

        renderSearchResults(input.value);
    });

    input.addEventListener('focus', () => {
        renderSearchResults(input.value);
    });

    clearButton.addEventListener('click', () => {
        selectClusterPlayer('all');
        input.focus();
    });

    document.addEventListener('click', event => {
        if (!event.target.closest('.player-search-wrapper')) {
            closeResults();
        }
    });
}

function getSelectedPlayerData() {
    const selected = selectedClusterPlayer || 'all';
    const players = clusteringDashboard?.players || [];

    if (selected === 'all') {
        return null;
    }

    return players.find(
        player => player.player_name === selected
    ) || null;
}

function refreshClusteringCharts() {
    if (!clusteringDashboard) return;

    const selectedPlayer = getSelectedPlayerData();
    renderPlayerSummary(selectedPlayer);
    renderClusterScatter(selectedPlayer);
    renderClusterDistribution();
    renderClusterTrend();
    renderClusterRadar(selectedPlayer);
}

function renderPlayerSummary(player) {
    const container = document.getElementById('player-summary');
    if (!container) return;

    if (!player) {
        container.innerHTML = `
            <div class="player-summary-card"><h4>Filtro atual</h4><div class="summary-value">Todos</div><div class="cluster-meta-subtitle">Os gráficos mostram o conjunto completo</div></div>
            <div class="player-summary-card"><h4>Jogadores no dashboard</h4><div class="summary-value">${clusteringDashboard.unique_players ?? '--'}</div><div class="cluster-meta-subtitle">base total carregada</div></div>
            <div class="player-summary-card"><h4>Clusters</h4><div class="summary-value">${clusteringDashboard.cluster_count ?? '--'}</div><div class="cluster-meta-subtitle">${clusteringDashboard.top_style || '--'} lidera a distribuição</div></div>
        `;
        return;
    }

    const clustersSeen = Array.isArray(player.clusters) ? player.clusters.join(', ') : '--';
    container.innerHTML = `
        <div class="player-summary-card"><h4>Jogador selecionado</h4><div class="summary-value">${player.player_name || '--'}</div><div class="cluster-meta-subtitle">${player.rows ?? '--'} registos · ${player.seasons ?? '--'} temporadas</div></div>
        <div class="player-summary-card"><h4>Tipo de jogador</h4><div class="summary-value">${player.style_label || '--'}</div><div class="cluster-meta-subtitle">cluster dominante: ${player.dominant_cluster ?? '--'}</div></div>
        <div class="player-summary-card"><h4>Superfície favorita</h4><div class="summary-value">${player.favorite_surface || '--'}</div><div class="cluster-meta-subtitle">clusters vistos: ${clustersSeen}</div></div>
        <div class="player-summary-card"><h4>Win rate</h4><div class="summary-value">${player.win_rate != null ? `${Number(player.win_rate * 100).toFixed(1)}%` : '--'}</div><div class="cluster-meta-subtitle">média nos dados filtrados</div></div>
    `;
}

function renderClusterScatter(player) {
    const canvas = document.getElementById('clusterScatterChart');
    if (!canvas) return;

    const points = clusteringDashboard.scatter_points || [];
    const selectedName = player?.player_name || null;
    const allPoints = points.map(point => ({
        x: Number(point.x) || 0,
        y: Number(point.y) || 0,
        player_name: point.player_name,
        cluster: point.cluster,
        season: point.season,
        style_label: point.style_label,
        favorite_surface: point.favorite_surface,
        win_rate: Number(point.win_rate) || 0
    }));

    const baseDataset = selectedName
        ? allPoints.filter(point => point.player_name !== selectedName)
        : allPoints;
    const highlightDataset = selectedName
        ? allPoints.filter(point => point.player_name === selectedName)
        : [];

    if (clusterScatterChart) clusterScatterChart.destroy();
    clusterScatterChart = new Chart(canvas.getContext('2d'), {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: selectedName ? 'Outros jogadores' : 'Jogadores',
                    data: baseDataset,
                    backgroundColor: 'rgba(164, 102, 234, 0.35)',
                    borderColor: 'rgba(164, 102, 234, 0.35)',
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: selectedName || 'Destaque',
                    data: highlightDataset,
                    backgroundColor: '#ffb86b',
                    borderColor: '#ffb86b',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: '#cfcbe6' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                y: { ticks: { color: '#cfcbe6' }, grid: { color: 'rgba(255,255,255,0.1)' } }
            },
            plugins: {
                legend: { labels: { color: '#cfcbe6' } },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const raw = context.raw || {};
                            return `${raw.player_name || 'Jogador'} | C${raw.cluster ?? '-'} | ${raw.style_label || '-'} | ${raw.favorite_surface || '-'}`;
                        }
                    }
                }
            }
        }
    });
}

function renderClusterDistribution() {
    const canvas = document.getElementById('clusterDistributionChart');
    if (!canvas) return;

    const clusters = (clusteringDashboard.cluster_distribution || []).slice().sort((a, b) => a.cluster - b.cluster);
    if (clusterDistributionChart) clusterDistributionChart.destroy();
    clusterDistributionChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: clusters.map(item => `C${item.cluster}`),
            datasets: [{
                label: 'Jogadores',
                data: clusters.map(item => item.size),
                backgroundColor: ['#667eea', '#c084fc', '#a466ea', '#ffb86b', '#00c853'],
                borderRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label(context) { return `${context.parsed.y} jogadores`; } } }
            },
            scales: {
                y: { beginAtZero: true, ticks: { color: '#cfcbe6' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { ticks: { color: '#cfcbe6' }, grid: { display: false } }
            }
        }
    });
}

function renderClusterTrend() {
    const canvas = document.getElementById('clusterTrendChart');
    if (!canvas) return;

    const trend = clusteringDashboard.season_trend || [];
    const seasons = Array.from(new Set(trend.map(item => item.season))).sort((a, b) => a - b);
    const clusters = Array.from(new Set(trend.map(item => item.cluster))).sort((a, b) => a - b);
    const datasets = clusters.map((cluster, index) => ({
        label: `C${cluster}`,
        data: seasons.map(season => {
            const match = trend.find(item => item.season === season && item.cluster === cluster);
            return match ? match.count : 0;
        }),
        borderColor: ['#667eea', '#ffb86b', '#a466ea', '#00c853', '#ff5252'][index % 5],
        backgroundColor: 'transparent',
        tension: 0.35,
        fill: false
    }));

    if (clusterTrendChart) clusterTrendChart.destroy();
    clusterTrendChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: seasons, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { ticks: { color: '#cfcbe6' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { ticks: { color: '#cfcbe6' }, grid: { color: 'rgba(255,255,255,0.1)' } }
            },
            plugins: {
                legend: { labels: { color: '#cfcbe6' } }
            }
        }
    });
}

function renderClusterRadar(player) {
    const canvas = document.getElementById('clusterRadarChart');
    if (!canvas) return;

    const clusterId = player?.dominant_cluster ?? (clusteringDashboard.cluster_distribution?.[0]?.cluster ?? 0);
    const profile = clusteringDashboard.radar_profiles?.[clusterId] || clusteringDashboard.radar_profiles?.[String(clusterId)] || null;
    if (!profile) return;

    if (clusterRadarChart) clusterRadarChart.destroy();
    clusterRadarChart = new Chart(canvas.getContext('2d'), {
        type: 'radar',
        data: {
            labels: profile.labels || [],
            datasets: [{
                label: player?.player_name || profile.style_label || 'Cluster',
                data: profile.values || [],
                borderColor: '#ffb86b',
                backgroundColor: 'rgba(255, 184, 107, 0.2)',
                pointBackgroundColor: '#ffb86b',
                pointBorderColor: '#ffffff',
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.15)' },
                    grid: { color: 'rgba(255,255,255,0.15)' },
                    pointLabels: { color: '#cfcbe6' },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { labels: { color: '#cfcbe6' } }
            }
        }
    });
}

// ==================== HISTÓRICO (TRACK RECORD) ====================
// Uma linha por previsão avaliada: {date, tourney, surface, p1, p2, pick, winner, score,
// conf, correct, odd, stake (fração da banca), profit (fração da banca)}.
let trackRecord = [];
let trBankrollChart = null;
let trCalibrationChart = null;
const TR_PAGE_SIZE = 50;
let trVisible = TR_PAGE_SIZE;

const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const signedPct = (value, digits = 2) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`;
const formatDatePt = iso => {
    const [y, m, d] = String(iso || '').split('-');
    return d ? `${d}/${m}/${y}` : String(iso || '');
};

async function loadTrackRecord() {
    if (window.trLoading) return;
    window.trLoading = true;
    setTabStatus('tab-historico', 'loading', 'A carregar o histórico de previsões…');
    try {
        const data = await getJSON('analytics/track_record.json');
        trackRecord = Array.isArray(data) ? data : [];
        populateTrackFilters();
        bindTrackFilters();
        renderTrackRecord();
        window.trackLoaded = true;
        if (trackRecord.length) {
            setTabStatus('tab-historico', 'ok');
        } else {
            setTabStatus('tab-historico', 'empty', 'Ainda não há previsões avaliadas. O histórico enche-se com as comparações diárias.');
        }
    } catch (error) {
        console.error('Erro ao carregar histórico:', error);
        setTabStatus('tab-historico', 'error',
            'Não foi possível carregar o histórico. Verifique a ligação e tente de novo.', loadTrackRecord);
        window.trackLoaded = false;
    } finally {
        window.trLoading = false;
    }
}

function populateTrackFilters() {
    const fill = (selectId, values) => {
        const select = document.getElementById(selectId);
        select.querySelectorAll('option:not([value="all"])').forEach(o => o.remove());
        values.forEach(value => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            select.appendChild(option);
        });
    };
    const uniq = key => [...new Set(trackRecord.map(r => r[key]).filter(Boolean))].sort((a, b) => a.localeCompare(b));
    fill('tr-tourney', uniq('tourney'));
    fill('tr-surface', uniq('surface'));
}

function bindTrackFilters() {
    if (window.trFiltersBound) return;
    window.trFiltersBound = true;
    ['tr-period', 'tr-tourney', 'tr-surface', 'tr-conf', 'tr-bets'].forEach(id =>
        document.getElementById(id).addEventListener('change', () => { trVisible = TR_PAGE_SIZE; renderTrackRecord(); }));
    document.getElementById('tr-player').addEventListener('input', () => { trVisible = TR_PAGE_SIZE; renderTrackRecord(); });
    document.getElementById('tr-more').addEventListener('click', () => { trVisible += TR_PAGE_SIZE; renderTrackTable(filteredTrackRecord()); });
}

function filteredTrackRecord() {
    const period = document.getElementById('tr-period').value;
    const tourney = document.getElementById('tr-tourney').value;
    const surface = document.getElementById('tr-surface').value;
    const minConf = Number(document.getElementById('tr-conf').value) || 0;
    const player = document.getElementById('tr-player').value.trim().toLowerCase();
    const onlyBets = document.getElementById('tr-bets').checked;
    // "Últimos N dias" contam a partir da última previsão avaliada (os dados podem não ser de hoje)
    const lastDate = trackRecord.reduce((max, r) => (r.date > max ? r.date : max), '');
    const cutoff = period === 'all' || !lastDate ? '' :
        new Date(new Date(lastDate).getTime() - Number(period) * 86400000).toISOString().slice(0, 10);

    return trackRecord.filter(r =>
        (!cutoff || r.date > cutoff) &&
        (tourney === 'all' || r.tourney === tourney) &&
        (surface === 'all' || r.surface === surface) &&
        (Number(r.conf) || 0) >= minConf &&
        (!onlyBets || r.stake > 0) &&
        (!player || `${r.p1} ${r.p2}`.toLowerCase().includes(player))
    );
}

function trackSummary(rows) {
    const bets = rows.filter(r => r.stake > 0).sort((a, b) => a.date.localeCompare(b.date));
    const staked = bets.reduce((sum, r) => sum + r.stake, 0);
    let cumulative = 0, peak = 0, maxDrawdown = 0;
    const curve = new Map();
    bets.forEach(r => {
        cumulative += Number(r.profit) || 0;
        peak = Math.max(peak, cumulative);
        maxDrawdown = Math.max(maxDrawdown, peak - cumulative);
        curve.set(r.date, cumulative);
    });
    return {
        total: rows.length,
        correct: rows.filter(r => r.correct).length,
        bets: bets.length,
        staked,
        profit: cumulative,
        roi: staked > 0 ? cumulative / staked : null,
        maxDrawdown,
        curve,
    };
}

function renderTrackRecord() {
    const rows = filteredTrackRecord();
    const s = trackSummary(rows);
    const setKpi = (id, text, cls = '') => {
        const el = document.getElementById(id);
        el.textContent = text;
        el.classList.toggle('stat-positive', cls === 'pos');
        el.classList.toggle('stat-negative', cls === 'neg');
    };
    const tone = v => (v > 0 ? 'pos' : v < 0 ? 'neg' : '');
    setKpi('tr-kpi-total', String(s.total));
    setKpi('tr-kpi-acc', s.total ? pct(s.correct / s.total) : '--');
    setKpi('tr-kpi-bets', String(s.bets));
    setKpi('tr-kpi-roi', s.roi === null ? '--' : signedPct(s.roi, 1), s.roi === null ? '' : tone(s.roi));
    setKpi('tr-kpi-profit', s.bets ? signedPct(s.profit) : '--', s.bets ? tone(s.profit) : '');
    setKpi('tr-kpi-drawdown', s.bets ? `-${(s.maxDrawdown * 100).toFixed(2)}%` : '--');

    renderTrackBankroll(s);
    renderTrackCalibration(rows);
    renderTrackTable(rows);
}

const chartAxis = { ticks: { color: '#cfcbe6' }, grid: { color: 'rgba(255,255,255,0.08)' } };

function renderTrackBankroll(summary) {
    const canvas = document.getElementById('trBankrollChart');
    const empty = document.getElementById('tr-bankroll-empty');
    if (trBankrollChart) { trBankrollChart.destroy(); trBankrollChart = null; }
    const hasBets = summary.bets > 0;
    canvas.hidden = !hasBets;
    empty.hidden = hasBets;
    if (!hasBets || typeof Chart === 'undefined') return;
    const labels = [...summary.curve.keys()];
    trBankrollChart = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels.map(formatDatePt),
            datasets: [{
                label: 'Lucro acumulado (% da banca)',
                data: labels.map(d => Number((summary.curve.get(d) * 100).toFixed(2))),
                borderColor: '#a78bfa',
                backgroundColor: 'rgba(167, 139, 250, 0.15)',
                fill: true,
                tension: 0.25,
                pointRadius: 2,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#cfcbe6' } } },
            scales: { x: chartAxis, y: { ...chartAxis, ticks: { ...chartAxis.ticks, callback: v => `${v}%` } } }
        }
    });
}

function renderTrackCalibration(rows) {
    const canvas = document.getElementById('trCalibrationChart');
    if (trCalibrationChart) { trCalibrationChart.destroy(); trCalibrationChart = null; }
    if (typeof Chart === 'undefined') return;
    const bins = [[50, 60], [60, 70], [70, 80], [80, 90], [90, 101]];
    const stats = bins.map(([lo, hi]) => {
        const inBin = rows.filter(r => r.conf >= lo && r.conf < hi);
        return {
            label: `${lo}–${Math.min(hi, 100)}%`,
            n: inBin.length,
            predicted: inBin.length ? inBin.reduce((sum, r) => sum + r.conf, 0) / inBin.length : null,
            actual: inBin.length ? (inBin.filter(r => r.correct).length / inBin.length) * 100 : null,
        };
    });
    trCalibrationChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: stats.map(s => `${s.label} (${s.n})`),
            datasets: [
                { label: 'Confiança média prevista', data: stats.map(s => s.predicted), backgroundColor: 'rgba(167, 139, 250, 0.35)', borderColor: '#a78bfa', borderWidth: 1 },
                { label: 'Acerto real', data: stats.map(s => s.actual), backgroundColor: 'rgba(74, 222, 128, 0.55)', borderColor: '#4ade80', borderWidth: 1 }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#cfcbe6' } },
                tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y === null ? '—' : ctx.parsed.y.toFixed(1) + '%'}` } }
            },
            scales: { x: chartAxis, y: { ...chartAxis, min: 0, max: 100, ticks: { ...chartAxis.ticks, callback: v => `${v}%` } } }
        }
    });
}

function renderTrackTable(rows) {
    const body = document.getElementById('tr-body');
    const sorted = [...rows].sort((a, b) => b.date.localeCompare(a.date));
    const visible = sorted.slice(0, trVisible);
    body.textContent = '';
    const cell = (text, className = '') => {
        const td = document.createElement('td');
        td.textContent = text;
        if (className) td.className = className;
        return td;
    };
    visible.forEach(r => {
        const tr = document.createElement('tr');
        tr.appendChild(cell(formatDatePt(r.date), 'tr-nowrap'));
        tr.appendChild(cell([r.tourney, r.surface].filter(Boolean).join(' · ')));
        tr.appendChild(cell(`${r.p1} vs ${r.p2}`));
        tr.appendChild(cell(`${r.pick}${r.conf ? ` (${Number(r.conf).toFixed(1)}%)` : ''}`));
        const result = cell(`${r.correct ? '✓ Acertou' : '✗ Falhou'}${r.score ? ` · ${r.score}` : ''}`, r.correct ? 'tr-ok' : 'tr-fail');
        tr.appendChild(result);
        tr.appendChild(cell(r.odd ? Number(r.odd).toFixed(2) : '—', 'tr-num'));
        tr.appendChild(cell(r.stake > 0 ? pct(r.stake) : '—', 'tr-num'));
        tr.appendChild(cell(r.stake > 0 ? signedPct(Number(r.profit) || 0) : '—',
            `tr-num ${r.stake > 0 ? ((r.profit || 0) > 0 ? 'tr-ok' : 'tr-fail') : ''}`));
        body.appendChild(tr);
    });
    document.getElementById('tr-count').textContent = rows.length
        ? `A mostrar ${visible.length} de ${rows.length} previsões`
        : 'Nenhuma previsão corresponde aos filtros.';
    document.getElementById('tr-more').hidden = visible.length >= sorted.length;
}

// Pequena espera mínima só para evitar um "flash" do ecrã de loading
// quando tudo carrega quase instantaneamente.
const minDelay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Se por algum motivo os dados nunca chegarem (rede em baixo, etc.),
// não bloqueia o site para sempre - avança ao fim de 8s na mesma.
const readyOrTimeout = Promise.race([
    window.predictionsReady || Promise.resolve(),
    minDelay(8000)
]);

const windowLoaded = new Promise(resolve => {
    if (document.readyState === 'complete') resolve();
    else window.addEventListener('load', resolve, { once: true });
});

Promise.all([windowLoaded, readyOrTimeout, minDelay(2000)]).then(() => {
    const loadingScreen = document.getElementById('loadingScreen');
    if (!loadingScreen) return;

    loadingScreen.addEventListener('animationend', () => loadingScreen.remove(), { once: true });
    loadingScreen.classList.add('is-hidden');

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        loadingScreen.remove();
    }
});

// ==================== APP (PWA) ====================
// Link de convite do canal do Telegram (privado). Vazio = botão escondido.
const TELEGRAM_INVITE_URL = 'https://t.me/+d6AKCEp2-9E2NTI0';

// Abrir o separador indicado no URL (#previsoes, #estatisticas, #historico, #clustering)
function openTabFromHash() {
    const name = location.hash.replace('#', '');
    const tab = document.getElementById(`tabbtn-${name}`);
    if (tab && tab.getAttribute('aria-selected') !== 'true') tab.click();
}
openTabFromHash();
window.addEventListener('hashchange', openTabFromHash);

const isStandalone = () =>
    window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true;
const isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

// Telegram
const telegramBtn = document.getElementById('telegramBtn');
if (TELEGRAM_INVITE_URL) {
    telegramBtn.href = TELEGRAM_INVITE_URL;
    telegramBtn.setAttribute('aria-label', 'Receber as previsões no canal do Telegram (abre noutra janela)');
    telegramBtn.hidden = false;
}

// Instalar a app: Android/desktop (Chrome) pelo beforeinstallprompt; iPhone com instruções
const installBtn = document.getElementById('installBtn');
const iosInstallBtn = document.getElementById('iosInstallBtn');
const iosDialog = document.getElementById('iosInstallDialog');
let deferredInstallPrompt = null;

window.addEventListener('beforeinstallprompt', event => {
    event.preventDefault();
    deferredInstallPrompt = event;
    if (!isStandalone()) installBtn.hidden = false;
});

installBtn.addEventListener('click', async () => {
    if (!deferredInstallPrompt) return;
    deferredInstallPrompt.prompt();
    await deferredInstallPrompt.userChoice;
    deferredInstallPrompt = null;
    installBtn.hidden = true;
});

window.addEventListener('appinstalled', () => {
    installBtn.hidden = true;
    deferredInstallPrompt = null;
});

if (isIOS && !isStandalone()) {
    iosInstallBtn.hidden = false;
    iosInstallBtn.addEventListener('click', () => {
        if (typeof iosDialog.showModal === 'function') iosDialog.showModal();
        else iosDialog.setAttribute('open', '');
    });
}

// Aviso de "sem ligação"
const offlineBanner = document.getElementById('offlineBanner');
const updateOnlineStatus = () => { offlineBanner.hidden = navigator.onLine; };
window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);
updateOnlineStatus();

// Service worker: cache para abrir offline e aviso quando há uma versão nova do site
if ('serviceWorker' in navigator) {
    const updateToast = document.getElementById('updateToast');
    let updateRequested = false;

    const showUpdate = worker => {
        updateToast.hidden = false;
        document.getElementById('updateBtn').onclick = () => {
            updateRequested = true;
            worker.postMessage('SKIP_WAITING');
        };
    };

    // Só recarrega quando o utilizador pediu a versão nova (na 1.ª visita o
    // service worker também passa a controlar a página, e aí não se recarrega)
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (!updateRequested) return;
        updateRequested = false;
        location.reload();
    });

    window.addEventListener('load', async () => {
        try {
            const registration = await navigator.serviceWorker.register('sw.js');
            if (registration.waiting && navigator.serviceWorker.controller) showUpdate(registration.waiting);
            registration.addEventListener('updatefound', () => {
                const worker = registration.installing;
                worker?.addEventListener('statechange', () => {
                    // só há "versão nova" se já havia um service worker a controlar a página
                    if (worker.state === 'installed' && navigator.serviceWorker.controller) showUpdate(worker);
                });
            });
        } catch (error) {
            console.warn('Service worker não registado:', error);
        }
    });
}
