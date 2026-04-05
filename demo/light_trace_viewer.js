const state = { episodes: [], selectedPath: null, selectedTrajectory: 0 };

function formatScores(values) {
  if (!values || !values.length) return "-";
  return values.map(v => Number(v).toFixed(3)).join(", ");
}

function renderEpisodeList() {
  const el = document.getElementById("episodeList");
  if (!state.episodes.length) {
    el.innerHTML = `<div class="empty-state">No trace files found.</div>`;
    return;
  }
  el.innerHTML = state.episodes.map(ep => `
    <button class="episode-item ${state.selectedPath === ep.path ? "active" : ""}" data-path="${ep.path}">
      <div class="episode-title">${ep.qid}</div>
      <div class="episode-meta">${ep.status} · v${ep.version}</div>
      <div class="episode-question">${ep.question || ""}</div>
      <div class="episode-scores">raw: ${formatScores(ep.raw_scores)}</div>
      <div class="episode-scores">norm: ${formatScores(ep.normalized_scores)}</div>
      ${ep.reason ? `<div class="episode-reason">reason: ${ep.reason}</div>` : ""}
    </button>
  `).join("");
  document.querySelectorAll(".episode-item").forEach(btn => {
    btn.addEventListener("click", async () => {
      state.selectedPath = btn.dataset.path;
      state.selectedTrajectory = 0;
      renderEpisodeList();
      await loadEpisode();
    });
  });
}

function renderSummary(data) {
  const el = document.getElementById("episodeSummary");
  el.className = "episode-summary";
  el.innerHTML = `
    <h2>${data.qid}</h2>
    <div class="summary-grid">
      <div><strong>Status</strong><span>${data.status}</span></div>
      <div><strong>Version</strong><span>${data.version}</span></div>
      <div><strong>Reason</strong><span>${data.reason || "-"}</span></div>
      <div><strong>Ground Truth</strong><span>${data.ground_truth || "-"}</span></div>
      <div><strong>Raw Scores</strong><span>${formatScores(data.raw_scores)}</span></div>
      <div><strong>Normalized</strong><span>${formatScores(data.normalized_scores)}</span></div>
    </div>
    <div class="question-block">
      <strong>Question</strong>
      <pre>${data.question || ""}</pre>
    </div>
  `;
}

function renderTrajectoryTabs(data) {
  const tabs = document.getElementById("trajectoryTabs");
  const detail = document.getElementById("trajectoryDetail");
  if (!data.trajectories || !data.trajectories.length) {
    tabs.innerHTML = "";
    detail.innerHTML = `<div class="empty-state">No trajectory details for this trace.</div>`;
    return;
  }
  tabs.innerHTML = data.trajectories.map((traj, idx) => `
    <button class="traj-tab ${idx === state.selectedTrajectory ? "active" : ""}" data-idx="${idx}">
      traj ${idx} · score ${Number(traj.final_score || 0).toFixed(3)}
    </button>
  `).join("");
  document.querySelectorAll(".traj-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      state.selectedTrajectory = Number(btn.dataset.idx);
      renderTrajectoryTabs(data);
      renderTrajectoryDetail(data.trajectories[state.selectedTrajectory]);
    });
  });
  renderTrajectoryDetail(data.trajectories[state.selectedTrajectory]);
}

function renderTrajectoryDetail(traj) {
  const detail = document.getElementById("trajectoryDetail");
  const steps = (traj.step_events || []).map(step => `
    <div class="step-card">
      <div class="step-header">
        <span>turn ${step.turn}</span>
        <span>in ${step.input_tokens} tok</span>
        <span>out ${step.output_tokens} tok</span>
      </div>
      <div class="step-row"><strong>tool</strong><code>${step.tool_call || "-"}</code></div>
      <div class="step-row"><strong>type</strong><span>${step.tool_result_type || "-"}</span></div>
      <div class="step-row"><strong>tool preview</strong><pre>${step.tool_result_preview || ""}</pre></div>
      <div class="step-row"><strong>completion</strong><pre>${step.completion_text || ""}</pre></div>
    </div>
  `).join("");
  detail.innerHTML = `
    <div class="summary-grid">
      <div><strong>Final Score</strong><span>${Number(traj.final_score || 0).toFixed(3)}</span></div>
      <div><strong>Normalized</strong><span>${Number(traj.normalized_score || 0).toFixed(3)}</span></div>
      <div><strong>Base Score</strong><span>${Number(traj.base_score || 0).toFixed(3)}</span></div>
      <div><strong>Terminated</strong><span>${traj.terminated_reason || "-"}</span></div>
      <div><strong>Search Calls</strong><span>${traj.search_calls || 0}</span></div>
      <div><strong>Access Calls</strong><span>${traj.access_calls || 0}</span></div>
      <div><strong>Repeats</strong><span>${traj.repeat_calls || 0}</span></div>
      <div><strong>LLM Turns</strong><span>${traj.llm_turns || 0}</span></div>
    </div>
    <div class="penalty-box">
      <strong>Penalty Breakdown</strong>
      <pre>${JSON.stringify(traj.penalty_breakdown || {}, null, 2)}</pre>
    </div>
    <div class="answer-box">
      <strong>Final Answer</strong>
      <pre>${traj.final_answer || ""}</pre>
    </div>
    <div class="steps">${steps || '<div class="empty-state">No step events captured.</div>'}</div>
  `;
}

async function loadEpisode() {
  if (!state.selectedPath) return;
  const res = await fetch(`/api/episode?path=${encodeURIComponent(state.selectedPath)}`);
  const data = await res.json();
  renderSummary(data);
  renderTrajectoryTabs(data);
}

async function refreshEpisodes() {
  const res = await fetch("/api/episodes");
  const data = await res.json();
  state.episodes = data.episodes || [];
  if (!state.selectedPath && state.episodes.length) {
    state.selectedPath = state.episodes[0].path;
  }
  renderEpisodeList();
  if (state.selectedPath) {
    await loadEpisode();
  }
}

document.getElementById("refreshButton").addEventListener("click", refreshEpisodes);
refreshEpisodes();
setInterval(refreshEpisodes, 5000);
