document.addEventListener("DOMContentLoaded", () => {
    // 1. API Endpoints
    const DATASET_API = "/api/dataset";
    const ANSWER_API = "/api/answer";
    const METRICS_API = "/api/metrics";

    // 2. DOM Elements
    const scenarioList = document.getElementById("scenario-list");
    const passageInput = document.getElementById("passage-input");
    const questionInput = document.getElementById("question-input");
    const askBtn = document.getElementById("ask-btn");

    const emValue = document.querySelector("#kpi-em .kpi-value");
    const f1Value = document.querySelector("#kpi-f1 .kpi-value");
    const latencyValue = document.querySelector("#kpi-latency .kpi-value");

    const outputBox = document.getElementById("output-box");
    const placeholderText = outputBox.querySelector(".placeholder-text");
    const outputContent = outputBox.querySelector(".output-content");
    const answerScore = document.getElementById("answer-score");
    const answerText = document.getElementById("answer-text");

    const textWords = document.getElementById("text-words");
    const textSentences = document.getElementById("text-sentences");
    const textChars = document.getElementById("text-chars");
    const textReadability = document.getElementById("text-readability");
    const categoryList = document.getElementById("category-stats-list");

    let dataset = [];

    // 3. Load initial statistics and scenarios
    loadMetrics();
    loadDataset();

    // 4. Input listener for live word/sentence counting
    passageInput.addEventListener("input", updateLocalTextProfile);

    // 5. Button click listener to fetch prediction
    askBtn.addEventListener("click", queryModelAnswer);

    // --- FETCH SCENARIOS FROM DATASET ---
    async function loadDataset() {
        try {
            const response = await fetch(DATASET_API);
            if (!response.ok) throw new Error("Dataset fetch failed");
            dataset = await response.json();

            scenarioList.innerHTML = "";
            dataset.forEach((scenario, index) => {
                const btn = document.createElement("button");
                btn.className = "scenario-btn";
                btn.textContent = `${scenario.category} (S${scenario.id})`;
                btn.addEventListener("click", () => {
                    // Highlight selected scenario
                    document.querySelectorAll(".scenario-btn").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");

                    // Fill inputs
                    passageInput.value = scenario.passage;
                    questionInput.value = scenario.qas[0].question;

                    // Reset output layout
                    outputContent.classList.add("hidden");
                    placeholderText.classList.remove("hidden");
                    updateLocalTextProfile();
                });
                scenarioList.appendChild(btn);
            });

            // Auto-click first scenario on startup
            if (dataset.length > 0) scenarioList.querySelector(".scenario-btn").click();
        } catch (error) {
            scenarioList.innerHTML = `<span style="color: var(--accent-amber)">Error: Check backend server connection.</span>`;
        }
    }

    // --- FETCH ACCURACY AND LATENCY STATS ---
    async function loadMetrics() {
        try {
            const response = await fetch(METRICS_API);
            if (!response.ok) throw new Error("Metrics fetch failed");
            const data = await response.json();

            // Set values in KPI cards
            emValue.textContent = `${(data.summary.avg_em * 100).toFixed(1)}%`;
            f1Value.textContent = `${(data.summary.avg_f1 * 100).toFixed(1)}%`;
            latencyValue.textContent = `${data.summary.avg_latency.toFixed(3)}s`;

            // Render category progress lists
            categoryList.innerHTML = "";
            data.categories.forEach(cat => {
                const item = document.createElement("div");
                item.className = "category-stat-item";
                const f1Percent = (cat.f1 * 100).toFixed(0);

                item.innerHTML = `
                    <div class="category-stat-header">
                        <strong>${cat.category}</strong>
                        <span>F1: ${f1Percent}%</span>
                    </div>
                    <div class="cat-progress-container">
                        <div class="cat-progress-fill" style="width: ${f1Percent}%"></div>
                    </div>
                `;
                categoryList.appendChild(item);
            });
        } catch (error) {
            console.error("Failed to load validation metrics:", error);
        }
    }

    // --- QUERY MODEL FROM BACKEND ---
    async function queryModelAnswer() {
        const passage = passageInput.value.trim();
        const question = questionInput.value.trim();

        if (!passage || !question) return alert("Please fill all input fields!");

        askBtn.disabled = true;
        askBtn.textContent = "Processing...";

        try {
            const response = await fetch(ANSWER_API, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ passage, question })
            });

            if (!response.ok) throw new Error("API call error");
            const data = await response.json();

            // Toggle view elements
            placeholderText.classList.add("hidden");
            outputContent.classList.remove("hidden");

            const pred = data.prediction;
            answerScore.textContent = `${(pred.score * 100).toFixed(1)}%`;
            answerText.textContent = pred.answer;

            // Display text details
            if (data.analytics) {
                textWords.textContent = data.analytics.word_count;
                textSentences.textContent = data.analytics.sentence_count;
                textChars.textContent = data.analytics.char_count;
                textReadability.textContent = data.analytics.readability_score.toFixed(1);
            }
        } catch (error) {
            alert("Error: Check backend console logs!");
        } finally {
            askBtn.disabled = false;
            askBtn.textContent = "Get Answer";
        }
    }

    // --- HIGHLIGHT ANSWER WORD IN PASSAGE ---
    function createHtmlHighlight(passage, answer, start, end) {
        if (!answer) return escapeHtml(passage);

        // Exact location slice check
        if (start !== undefined && end !== undefined && start < end) {
            const slice = passage.substring(start, end);
            if (slice.toLowerCase() === answer.toLowerCase()) {
                return escapeHtml(passage.substring(0, start)) +
                    "<mark>" + escapeHtml(slice) + "</mark>" +
                    escapeHtml(passage.substring(end));
            }
        }

        // Substring fallback search
        const startIdx = passage.toLowerCase().indexOf(answer.toLowerCase());
        if (startIdx !== -1) {
            return escapeHtml(passage.substring(0, startIdx)) +
                "<mark>" + escapeHtml(passage.substring(startIdx, startIdx + answer.length)) + "</mark>" +
                escapeHtml(passage.substring(startIdx + answer.length));
        }

        return escapeHtml(passage) + `<br><br><strong>Extracted answer:</strong> <mark>${escapeHtml(answer)}</mark>`;
    }

    // --- CALC LOCAL TEXT STATS ---
    function updateLocalTextProfile() {
        const text = passageInput.value.trim();
        if (!text) {
            textWords.textContent = "-";
            textSentences.textContent = "-";
            textChars.textContent = "-";
            textReadability.textContent = "-";
            return;
        }

        const words = text.match(/\b\w+\b/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);

        textWords.textContent = words.length;
        textSentences.textContent = sentences.length;
        textChars.textContent = text.length;

        // ARI Formula: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
        const wCount = words.length;
        const sCount = sentences.length || 1;
        if (wCount > 0) {
            const ari = 4.71 * (text.length / wCount) + 0.5 * (wCount / sCount) - 21.43;
            textReadability.textContent = Math.max(1.0, ari).toFixed(1);
        }
    }

    // HTML escape wrapper
    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
