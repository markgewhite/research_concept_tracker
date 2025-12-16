/**
 * ArXiv Concept Tracker - Frontend Logic
 */

// API Configuration - use relative URL to work both locally and on HuggingFace
const API_BASE = '/api';

// Application State
let selectedSeeds = new Set();
const MAX_SEEDS = 5;

/**
 * Search for papers on ArXiv
 */
async function searchPapers() {
    const searchInput = document.getElementById('search-input');
    const query = searchInput.value.trim();

    if (!query) {
        showError('Please enter a search query');
        return;
    }

    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = '<div class="text-gray-600">Searching ArXiv...</div>';

    try {
        const response = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}&limit=20`);

        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }

        const data = await response.json();
        const papers = data.papers || [];

        if (papers.length === 0) {
            resultsDiv.innerHTML = '<div class="text-gray-600">No papers found. Try a different query.</div>';
            return;
        }

        renderSearchResults(papers);
        hideError();

    } catch (error) {
        console.error('Search error:', error);
        showError(`Search failed: ${error.message}`);
        resultsDiv.innerHTML = '';
    }
}

/**
 * Render search results as selectable cards
 */
function renderSearchResults(papers) {
    const resultsDiv = document.getElementById('search-results');

    resultsDiv.innerHTML = papers.map(paper => {
        const isSelected = selectedSeeds.has(paper.arxiv_id);
        const isDisabled = !isSelected && selectedSeeds.size >= MAX_SEEDS;

        const publishedDate = new Date(paper.published).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short'
        });

        return `
            <div class="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition ${isSelected ? 'bg-blue-50 border-blue-400' : ''}">
                <div class="flex items-start gap-3">
                    <input
                        type="checkbox"
                        id="seed-${paper.arxiv_id}"
                        ${isSelected ? 'checked' : ''}
                        ${isDisabled ? 'disabled' : ''}
                        onchange="toggleSeed('${paper.arxiv_id}', ${JSON.stringify(paper).replace(/"/g, '&quot;')})"
                        class="mt-1 w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                    >
                    <div class="flex-1">
                        <h3 class="font-semibold text-gray-800">${paper.title}</h3>
                        <p class="text-sm text-gray-600 mt-1">
                            ${paper.authors.slice(0, 3).join(', ')}${paper.authors.length > 3 ? ' et al.' : ''}
                        </p>
                        <div class="flex gap-4 mt-2 text-xs text-gray-500">
                            <span>${publishedDate}</span>
                            <span>${paper.arxiv_id}</span>
                            <a href="${paper.pdf_url}" target="_blank" class="text-blue-600 hover:underline">PDF</a>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Toggle seed paper selection
 */
function toggleSeed(arxivId, paperData) {
    const checkbox = document.getElementById(`seed-${arxivId}`);

    if (checkbox.checked) {
        if (selectedSeeds.size >= MAX_SEEDS) {
            checkbox.checked = false;
            showError(`Maximum ${MAX_SEEDS} seed papers allowed`);
            return;
        }
        selectedSeeds.add(arxivId);
        // Store paper data for display
        if (!window.seedPapers) window.seedPapers = {};
        window.seedPapers[arxivId] = paperData;
    } else {
        selectedSeeds.delete(arxivId);
    }

    updateSelectedSeeds();
    renderSearchResults(Object.values(window.seedPapers || {}));
}

/**
 * Update the selected seeds display
 */
function updateSelectedSeeds() {
    const seedsSection = document.getElementById('seeds-section');
    const selectedSeedsDiv = document.getElementById('selected-seeds');
    const seedCount = document.getElementById('seed-count');

    seedCount.textContent = selectedSeeds.size;

    if (selectedSeeds.size === 0) {
        seedsSection.style.display = 'none';
        return;
    }

    seedsSection.style.display = 'block';

    selectedSeedsDiv.innerHTML = Array.from(selectedSeeds).map(arxivId => {
        const paper = window.seedPapers[arxivId];
        return `
            <div class="flex items-center justify-between bg-blue-50 border border-blue-200 rounded-lg px-4 py-2">
                <div>
                    <span class="font-medium text-gray-800">${paper.title}</span>
                    <span class="text-sm text-gray-600 ml-2">(${paper.arxiv_id})</span>
                </div>
                <button
                    onclick="removeSeed('${arxivId}')"
                    class="text-red-600 hover:text-red-800 font-bold"
                >
                    ✕
                </button>
            </div>
        `;
    }).join('');
}

/**
 * Remove a seed paper
 */
function removeSeed(arxivId) {
    selectedSeeds.delete(arxivId);
    const checkbox = document.getElementById(`seed-${arxivId}`);
    if (checkbox) checkbox.checked = false;

    updateSelectedSeeds();
    renderSearchResults(Object.values(window.seedPapers || {}));
}

/**
 * Track concept evolution
 */
async function trackConcept() {
    if (selectedSeeds.size === 0) {
        showError('Please select at least one seed paper');
        return;
    }

    const endDate = document.getElementById('end-date').value;
    const windowMonths = parseInt(document.getElementById('window-months').value);
    const maxPapers = parseInt(document.getElementById('max-papers').value);

    if (!endDate) {
        showError('Please select an end date');
        return;
    }

    // Hide other sections and show loading
    document.getElementById('search-section').style.display = 'none';
    document.getElementById('seeds-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';

    const requestBody = {
        seed_paper_ids: Array.from(selectedSeeds),
        end_date: endDate,
        window_months: windowMonths,
        similarity_threshold: 0.50,
        max_papers_per_window: maxPapers
    };

    try {
        const response = await fetch(`${API_BASE}/track`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Tracking failed: ${response.statusText}`);
        }

        const data = await response.json();

        // Hide loading and show results
        document.getElementById('loading-section').style.display = 'none';
        document.getElementById('results-section').style.display = 'block';

        renderResults(data);
        hideError();

    } catch (error) {
        console.error('Tracking error:', error);
        showError(`Tracking failed: ${error.message}`);

        // Show search section again on error
        document.getElementById('loading-section').style.display = 'none';
        document.getElementById('search-section').style.display = 'block';
        document.getElementById('seeds-section').style.display = 'block';
    }
}

/**
 * Render tracking results
 */
function renderResults(data) {
    // Render summary stats
    const summaryDiv = document.getElementById('summary-stats');
    summaryDiv.innerHTML = `
        <div class="bg-blue-50 rounded-lg p-4 text-center">
            <div class="text-2xl font-bold text-blue-600">${data.num_steps}</div>
            <div class="text-sm text-gray-600">Time Windows</div>
        </div>
        <div class="bg-green-50 rounded-lg p-4 text-center">
            <div class="text-2xl font-bold text-green-600">${data.total_papers}</div>
            <div class="text-sm text-gray-600">Papers Tracked</div>
        </div>
        <div class="bg-purple-50 rounded-lg p-4 text-center">
            <div class="text-2xl font-bold text-purple-600">${(data.timeline.reduce((sum, step) => sum + step.avg_similarity, 0) / data.num_steps).toFixed(2)}</div>
            <div class="text-sm text-gray-600">Avg Similarity</div>
        </div>
        <div class="bg-orange-50 rounded-lg p-4 text-center">
            <div class="text-2xl font-bold text-orange-600">${data.window_months}m</div>
            <div class="text-sm text-gray-600">Window Size</div>
        </div>
    `;

    // Render timeline
    const timelineDiv = document.getElementById('timeline');
    timelineDiv.innerHTML = data.timeline.map(step => {
        const startDate = new Date(step.start_date).toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
        const endDate = new Date(step.end_date).toLocaleDateString('en-US', { year: 'numeric', month: 'short' });

        return `
            <div class="border border-gray-200 rounded-lg p-6 slide-in">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-xl font-semibold text-gray-800">
                        Step ${step.step_number}: ${startDate} → ${endDate}
                    </h3>
                    <span class="text-sm text-gray-600">${step.papers.length} papers</span>
                </div>

                <!-- Step Statistics -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <div class="bg-gray-50 rounded px-3 py-2">
                        <div class="text-sm text-gray-600">Avg Similarity</div>
                        <div class="text-lg font-semibold text-gray-800">${step.avg_similarity.toFixed(3)}</div>
                    </div>
                    <div class="bg-green-50 rounded px-3 py-2">
                        <div class="text-sm text-gray-600">High (≥0.70)</div>
                        <div class="text-lg font-semibold text-green-600">${step.num_high_confidence}</div>
                    </div>
                    <div class="bg-yellow-50 rounded px-3 py-2">
                        <div class="text-sm text-gray-600">Moderate (0.60-0.70)</div>
                        <div class="text-lg font-semibold text-yellow-600">${step.num_moderate}</div>
                    </div>
                    <div class="bg-orange-50 rounded px-3 py-2">
                        <div class="text-sm text-gray-600">Low (<0.60)</div>
                        <div class="text-lg font-semibold text-orange-600">${step.num_low}</div>
                    </div>
                </div>

                <!-- Papers -->
                <div class="space-y-2">
                    ${renderStepPapers(step.papers)}
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Render papers for a timeline step
 */
function renderStepPapers(papers) {
    // Sort by similarity descending
    const sortedPapers = [...papers].sort((a, b) => b.similarity - a.similarity);

    // Show top 10 papers by default
    const topPapers = sortedPapers.slice(0, 10);

    return topPapers.map((paper, index) => {
        const similarity = (paper.similarity || 0).toFixed(3);
        const similarityColor = paper.similarity >= 0.70 ? 'text-green-600' :
                                 paper.similarity >= 0.60 ? 'text-yellow-600' : 'text-orange-600';

        return `
            <div class="paper-card border border-gray-200 rounded-lg p-4 bg-white">
                <div class="flex items-start gap-3">
                    <div class="flex-shrink-0">
                        <div class="text-sm font-bold ${similarityColor}">${similarity}</div>
                    </div>
                    <div class="flex-1 min-w-0">
                        <h4 class="font-semibold text-gray-800 mb-1">${paper.title}</h4>
                        <p class="text-sm text-gray-600 mb-2">
                            ${paper.authors.slice(0, 3).join(', ')}${paper.authors.length > 3 ? ' et al.' : ''}
                        </p>
                        <details class="text-sm text-gray-700">
                            <summary class="cursor-pointer text-blue-600 hover:text-blue-800">View abstract</summary>
                            <p class="mt-2 text-gray-600">${paper.abstract}</p>
                        </details>
                        <div class="flex gap-3 mt-2 text-xs text-gray-500">
                            <span>${paper.arxiv_id}</span>
                            <a href="${paper.pdf_url}" target="_blank" class="text-blue-600 hover:underline">PDF</a>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Reset application to initial state
 */
function resetApp() {
    selectedSeeds.clear();
    window.seedPapers = {};

    document.getElementById('search-input').value = '';
    document.getElementById('search-results').innerHTML = '';
    document.getElementById('end-date').value = '2019-12-31';
    document.getElementById('window-months').value = '6';
    document.getElementById('max-papers').value = '500';

    document.getElementById('search-section').style.display = 'block';
    document.getElementById('seeds-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';

    hideError();
}

/**
 * Show error message
 */
function showError(message) {
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');

    errorMessage.textContent = message;
    errorSection.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(hideError, 5000);
}

/**
 * Hide error message
 */
function hideError() {
    document.getElementById('error-section').style.display = 'none';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('ArXiv Concept Tracker initialized');
    hideError();
});
