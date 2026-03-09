/* Memory Tab — 列表 + 搜索 + 高级筛选 + 详情 + 删除 + URL 状态 */

let memoryPage = 0;
let isSearchMode = false;
let currentQuery = '';

// ── Collection 下拉 ──

async function loadCollections() {
    try {
        const data = await API.get('/api/collections');
        const sel = document.getElementById('filter-collection');
        sel.innerHTML = '<option value="">All Collections</option>';
        if (data.collections) {
            for (const c of data.collections) {
                sel.innerHTML += `<option value="${esc(c.name)}">${esc(c.name)} (${c.count})</option>`;
            }
        }
    } catch (e) {
        console.error('loadCollections:', e);
    }
}

// ── Category tags 动态加载 ──

let _categoryTagsReady = Promise.resolve();

async function loadCategoryTags() {
    try {
        const data = await API.get('/api/categories');
        const el = document.getElementById('category-tags');
        el.innerHTML = '';
        if (data.categories) {
            for (const c of data.categories) {
                const btn = document.createElement('button');
                btn.className = 'tag-btn';
                btn.textContent = `${c.name} (${c.count})`;
                btn.dataset.category = c.name;
                btn.addEventListener('click', () => {
                    btn.classList.toggle('active');
                    applyFilters();
                });
                el.appendChild(btn);
            }
        }
    } catch (e) {
        console.error('loadCategoryTags:', e);
    }
}

// ── Perception type 动态加载 ──

function loadPerceptionTypes() {
    const types = ['visual', 'tactile', 'auditory', 'proprioceptive', 'procedural'];
    const sel = document.getElementById('filter-perception-type');
    sel.innerHTML = '<option value="">All</option>';
    for (const t of types) {
        sel.innerHTML += `<option value="${t}">${t}</option>`;
    }
}

// ── 高级搜索面板 ──

function initAdvancedSearch() {
    const panel = document.getElementById('advanced-search');
    const toggle = document.getElementById('advanced-toggle');

    // 默认折叠
    panel.style.display = 'none';

    toggle.addEventListener('click', () => {
        const visible = panel.style.display !== 'none';
        panel.style.display = visible ? 'none' : 'block';
        toggle.classList.toggle('active', !visible);
    });

    // Confidence 滑块
    const confMin = document.getElementById('filter-conf-min');
    const confMax = document.getElementById('filter-conf-max');
    const confMinVal = document.getElementById('conf-min-val');
    const confMaxVal = document.getElementById('conf-max-val');

    confMin.addEventListener('input', () => { confMinVal.textContent = confMin.value + '%'; });
    confMax.addEventListener('input', () => { confMaxVal.textContent = confMax.value + '%'; });
    confMin.addEventListener('change', applyFilters);
    confMax.addEventListener('change', applyFilters);

    // Time / Perception type 下拉
    document.getElementById('filter-days').addEventListener('change', applyFilters);
    document.getElementById('filter-perception-type').addEventListener('change', applyFilters);
}

// ── 收集筛选参数 ──

function getFilterParams() {
    const params = new URLSearchParams();

    const collection = document.getElementById('filter-collection').value;
    if (collection) params.set('collection', collection);

    const type = document.getElementById('filter-type').value;
    if (type) params.set('type', type);

    // 高级搜索参数
    const panel = document.getElementById('advanced-search');
    if (panel.style.display !== 'none') {
        // Category tags
        const activeTags = document.querySelectorAll('#category-tags .tag-btn.active');
        if (activeTags.length) {
            const cats = Array.from(activeTags).map(b => b.dataset.category);
            params.set('category', cats.join(','));
        }

        // Confidence
        const confMin = parseInt(document.getElementById('filter-conf-min').value);
        const confMax = parseInt(document.getElementById('filter-conf-max').value);
        if (confMin > 0) params.set('confidence_min', (confMin / 100).toFixed(2));
        if (confMax < 100) params.set('confidence_max', (confMax / 100).toFixed(2));

        // Days
        const days = document.getElementById('filter-days').value;
        if (days) params.set('days', days);

        // Perception type
        const pt = document.getElementById('filter-perception-type').value;
        if (pt) params.set('perception_type', pt);
    }

    return params;
}

function applyFilters() {
    loadMemories(0);
    syncURLState();
}

// ── Memory 列表 ──

async function loadMemories(page) {
    if (page !== undefined) memoryPage = page;
    isSearchMode = false;
    currentQuery = '';
    document.getElementById('clear-search').style.display = 'none';

    const el = document.getElementById('memory-list');
    el.innerHTML = '<div class="loading">Loading...</div>';

    const params = getFilterParams();
    params.set('page', memoryPage);
    params.set('limit', 30);

    try {
        const data = await API.get(`/api/memories?${params}`);

        if (!data.memories?.length) {
            el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">&#129302;</div><div class="empty-state-text">No memories yet. Use <code>learn</code> or <code>save_perception</code> MCP tools to create memories.</div></div>';
            document.getElementById('pagination').innerHTML = '';
            return;
        }

        renderMemoryList(el, data.memories);
        renderPagination('pagination', memoryPage, data.total, 30, loadMemories);
        syncURLState();
    } catch (e) {
        el.innerHTML = '<div class="empty-state"><div class="empty-state-text">Failed to load memories</div></div>';
    }
}

// ── 搜索 ──

async function searchMemories() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) { loadMemories(0); return; }

    isSearchMode = true;
    currentQuery = query;
    document.getElementById('clear-search').style.display = '';

    const el = document.getElementById('memory-list');
    el.innerHTML = '<div class="loading">Searching...</div>';

    const collection = document.getElementById('filter-collection').value;
    let url = `/api/search?q=${encodeURIComponent(query)}&top_k=30`;
    if (collection) url += `&collection=${encodeURIComponent(collection)}`;

    try {
        const data = await API.get(url);

        if (!data.results?.length) {
            el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">&#128269;</div><div class="empty-state-text">No results for "${esc(query)}"</div></div>`;
            document.getElementById('pagination').innerHTML = '';
            return;
        }

        renderMemoryList(el, data.results);
        document.getElementById('pagination').innerHTML = `<span class="page-info">${data.total} results</span>`;
        syncURLState();
    } catch (e) {
        el.innerHTML = '<div class="empty-state"><div class="empty-state-text">Search failed</div></div>';
    }
}

// ── 搜索高亮 ──

function highlightText(text, query) {
    if (!query) return esc(text);
    const escaped = esc(text);
    const terms = query.split(/\s+/).filter(Boolean);
    let result = escaped;
    for (const term of terms) {
        const re = new RegExp(
            `(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'
        );
        result = result.replace(re, '<mark>$1</mark>');
    }
    return result;
}

// ── 渲染记忆列表 ──

function renderMemoryList(el, memories) {
    el.innerHTML = memories.map(m => {
        const typeClass = m.type === 'fact' ? 'badge-fact' : 'badge-perception';
        const display = m.human_summary || m.content || '';
        const conf = m.confidence != null ? m.confidence : 0;
        const confPct = Math.round(conf * 100);

        // 搜索模式用高亮，浏览模式用普通 esc
        const displayHtml = isSearchMode ? highlightText(display, currentQuery) : esc(display);

        return `<div class="memory-item" onclick="showMemoryDetail(${m.id})">
            <div class="memory-item-header">
                <span class="memory-item-id">#${m.id}</span>
                <span class="badge ${typeClass}">${esc(m.type)}</span>
                ${m.perception_type ? `<span class="badge">${esc(m.perception_type)}</span>` : ''}
                <span class="badge">${esc(m.category || '')}</span>
            </div>
            <div class="memory-item-content">${displayHtml}</div>
            <div class="memory-item-meta">
                <span>${esc(m.collection || '')}</span>
                <span>Conf: ${confPct}%<span class="confidence-bar"><span class="confidence-fill" style="width:${confPct}%"></span></span></span>
                <span>${formatTime(m.created_at)}</span>
                ${m.access_count ? `<span>Accessed: ${m.access_count}</span>` : ''}
            </div>
        </div>`;
    }).join('');
}

// ── 详情 Modal ──

// ── Context JSON 分区渲染 ──

function renderContext(raw) {
    if (!raw) return '';
    try {
        const obj = JSON.parse(raw);
        if (typeof obj !== 'object' || obj === null) throw new Error('not object');

        // 四分区：params / spatial / robot / task + 其余 fallback
        const sections = ['params', 'spatial', 'robot', 'task'];
        let html = '<div class="context-sections">';
        const rendered = new Set();

        for (const key of sections) {
            if (obj[key] && typeof obj[key] === 'object') {
                html += renderContextSection(key, obj[key]);
                rendered.add(key);
            }
        }
        // fallback：未归类的顶层 key
        const remaining = Object.keys(obj).filter(k => !rendered.has(k));
        if (remaining.length) {
            const fallbackObj = {};
            for (const k of remaining) fallbackObj[k] = obj[k];
            html += renderContextSection('other', fallbackObj);
        }
        html += '</div>';
        return html;
    } catch {
        // JSON 解析失败 — fallback 为纯文本
        return `<div class="detail-content">${esc(raw)}</div>`;
    }
}

function renderContextSection(title, obj) {
    let html = `<div class="context-section">
        <div class="context-section-title">${esc(title)}</div>
        <div class="context-kv">`;

    for (const [k, v] of Object.entries(obj)) {
        const val = typeof v === 'object' ? JSON.stringify(v) : String(v);
        html += `<div class="context-kv-item">
            <span class="ck-key">${esc(k)}</span>
            <span class="ck-val">${esc(val)}</span>
        </div>`;
    }
    html += '</div></div>';
    return html;
}

// ── 详情 Modal（双视图 + Context 分区）──

async function showMemoryDetail(id) {
    const modal = document.getElementById('memory-modal');
    const body = document.getElementById('modal-body');
    document.getElementById('modal-id').textContent = id;

    body.innerHTML = '<div class="loading">Loading...</div>';
    modal.style.display = 'flex';

    try {
        const m = await API.get(`/api/memory/${id}`);
        if (m.error) {
            body.innerHTML = `<div class="empty-state"><div class="empty-state-text">${esc(m.error)}</div></div>`;
            return;
        }

        // 双视图：human_summary 默认展示，content 折叠
        const summaryRow = m.human_summary
            ? `<tr><td>Summary</td><td>${esc(m.human_summary)}</td></tr>`
            : '';

        const contentId = 'detail-content-' + id;
        const contentRow = m.content
            ? `<tr><td>Content</td><td>
                <button class="content-toggle" onclick="document.getElementById('${contentId}').classList.toggle('content-collapsed')">
                    Show / Hide Raw Content
                </button>
                <div id="${contentId}" class="detail-content content-collapsed">${esc(m.content)}</div>
               </td></tr>`
            : '';

        // Context 分区渲染
        const contextRow = m.context
            ? `<tr><td>Context</td><td>${renderContext(m.context)}</td></tr>`
            : '';

        body.innerHTML = `
            <table class="detail-table">
                <tr><td>Type</td><td><span class="badge ${m.type === 'fact' ? 'badge-fact' : 'badge-perception'}">${esc(m.type)}</span></td></tr>
                <tr><td>Collection</td><td>${esc(m.collection)}</td></tr>
                ${summaryRow}
                ${contentRow}
                ${m.perception_type ? `<tr><td>Perception Type</td><td>${esc(m.perception_type)}</td></tr>` : ''}
                <tr><td>Category</td><td>${esc(m.category)}</td></tr>
                <tr><td>Confidence</td><td>${m.confidence != null ? (m.confidence * 100).toFixed(1) + '%' : '-'}</td></tr>
                <tr><td>Decay Rate</td><td>${m.decay_rate != null ? m.decay_rate : '-'}</td></tr>
                <tr><td>Source</td><td>${esc(m.source)}</td></tr>
                <tr><td>Scope</td><td>${esc(m.scope)}</td></tr>
                <tr><td>Status</td><td>${esc(m.status)}</td></tr>
                <tr><td>Session ID</td><td>${esc(m.session_id || '-')}</td></tr>
                <tr><td>Access Count</td><td>${m.access_count || 0}</td></tr>
                <tr><td>Created</td><td>${formatTime(m.created_at)}</td></tr>
                <tr><td>Updated</td><td>${formatTime(m.updated_at)}</td></tr>
                ${contextRow}
                ${m.scope_files && m.scope_files !== '[]' ? `<tr><td>Scope Files</td><td>${esc(m.scope_files)}</td></tr>` : ''}
                ${m.scope_entities && m.scope_entities !== '[]' ? `<tr><td>Scope Entities</td><td>${esc(m.scope_entities)}</td></tr>` : ''}
            </table>
        `;

        // 绑定删除按钮
        document.getElementById('modal-delete').onclick = () => deleteMemory(id);
    } catch (e) {
        body.innerHTML = '<div class="empty-state"><div class="empty-state-text">Failed to load detail</div></div>';
    }
}

async function deleteMemory(id) {
    if (!confirm(`Delete memory #${id}?`)) return;

    try {
        const res = await API.del(`/api/memory/${id}`, { reason: 'Web UI delete' });
        if (res.error) {
            alert(res.error);
            return;
        }
        closeModal();
        if (isSearchMode) searchMemories();
        else loadMemories();
    } catch (e) {
        alert('Delete failed');
    }
}

function closeModal() {
    document.getElementById('memory-modal').style.display = 'none';
}

// ── URL 状态保存 ──

function syncURLState() {
    const params = new URLSearchParams();

    const q = document.getElementById('search-input').value.trim();
    if (q) params.set('q', q);

    const coll = document.getElementById('filter-collection').value;
    if (coll) params.set('collection', coll);

    const type = document.getElementById('filter-type').value;
    if (type) params.set('type', type);

    // 高级搜索参数
    const activeTags = document.querySelectorAll('#category-tags .tag-btn.active');
    if (activeTags.length) {
        params.set('category', Array.from(activeTags).map(b => b.dataset.category).join(','));
    }

    const confMin = parseInt(document.getElementById('filter-conf-min').value);
    if (confMin > 0) params.set('confidence_min', confMin);

    const confMax = parseInt(document.getElementById('filter-conf-max').value);
    if (confMax < 100) params.set('confidence_max', confMax);

    const days = document.getElementById('filter-days').value;
    if (days) params.set('days', days);

    const pt = document.getElementById('filter-perception-type').value;
    if (pt) params.set('perception_type', pt);

    const qs = params.toString();
    const newURL = qs ? `${location.pathname}?${qs}#memories` : `${location.pathname}#memories`;
    history.replaceState(null, '', newURL);
}

function restoreFromURL() {
    const params = new URLSearchParams(location.search);

    if (params.get('q')) {
        document.getElementById('search-input').value = params.get('q');
    }
    if (params.get('collection')) {
        document.getElementById('filter-collection').value = params.get('collection');
    }
    if (params.get('type')) {
        document.getElementById('filter-type').value = params.get('type');
    }

    // 高级搜索参数恢复
    const hasAdvanced = params.get('category') || params.get('confidence_min') ||
        params.get('confidence_max') || params.get('days') || params.get('perception_type');

    if (hasAdvanced) {
        document.getElementById('advanced-search').style.display = 'block';
        document.getElementById('advanced-toggle').classList.add('active');
    }

    if (params.get('confidence_min')) {
        const v = parseInt(params.get('confidence_min'));
        document.getElementById('filter-conf-min').value = v;
        document.getElementById('conf-min-val').textContent = v + '%';
    }
    if (params.get('confidence_max')) {
        const v = parseInt(params.get('confidence_max'));
        document.getElementById('filter-conf-max').value = v;
        document.getElementById('conf-max-val').textContent = v + '%';
    }
    if (params.get('days')) {
        document.getElementById('filter-days').value = params.get('days');
    }
    if (params.get('perception_type')) {
        document.getElementById('filter-perception-type').value = params.get('perception_type');
    }

    // Category tags 在 loadCategoryTags 后恢复
    const savedCats = params.get('category');
    if (savedCats) {
        // 等 category tags 加载完后恢复（由 _categoryTagsReady promise 控制）
        _categoryTagsReady.then(() => {
            const cats = savedCats.split(',');
            document.querySelectorAll('#category-tags .tag-btn').forEach(btn => {
                if (cats.includes(btn.dataset.category)) btn.classList.add('active');
            });
        });
    }

    // 如果有搜索参数，切到 memories tab 并搜索
    if (location.hash === '#memories' || params.get('q')) {
        // 切换 tab
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.querySelector('[data-tab="memories"]').classList.add('active');
        document.getElementById('tab-memories').classList.add('active');

        if (params.get('q')) {
            searchMemories();
        } else {
            loadMemories(0);
        }
    }
}

// ── 事件绑定 ──

document.addEventListener('DOMContentLoaded', () => {
    // 搜索
    document.getElementById('search-btn').addEventListener('click', searchMemories);
    document.getElementById('search-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') searchMemories();
    });
    document.getElementById('clear-search').addEventListener('click', () => {
        document.getElementById('search-input').value = '';
        loadMemories(0);
    });

    // 筛选变更
    document.getElementById('filter-collection').addEventListener('change', () => { applyFilters(); });
    document.getElementById('filter-type').addEventListener('change', () => { applyFilters(); });

    // Modal 关闭
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', closeModal);
    });
    document.getElementById('memory-modal').addEventListener('click', (e) => {
        if (e.target.id === 'memory-modal') closeModal();
    });

    // 高级搜索
    initAdvancedSearch();

    // 预加载数据
    loadCollections();
    _categoryTagsReady = loadCategoryTags();
    loadPerceptionTypes();

    // URL 状态恢复
    restoreFromURL();
});
