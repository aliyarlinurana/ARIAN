const riskColors = {
  Low: "#3FA773",
  Moderate: "#D8A31D",
  High: "#D96C3B",
  Extreme: "#B73333",
};

let forecast = [];
let metrics = {};
let selectedRegion = "Baku";
let selectedDate = "";
let map;
let markers = [];

const fmtPct = (value) => `${Math.round(value * 100)}%`;
const fmtOne = (value) => Number(value).toFixed(1);

async function loadData() {
  const [forecastResponse, metricsResponse] = await Promise.all([
    fetch("../outputs/forecast_30_days.json"),
    fetch("../outputs/metrics.json"),
  ]);
  forecast = await forecastResponse.json();
  metrics = await metricsResponse.json();
  selectedDate = forecast[0].date;
  selectedRegion = forecast[0].region;
  initializeControls();
  initializeMap();
  renderAll();
}

function initializeControls() {
  const dates = [...new Set(forecast.map((d) => d.date))];
  const regions = [...new Set(forecast.map((d) => d.region))].sort();

  document.getElementById("datePicker").min = dates[0];
  document.getElementById("datePicker").max = dates[dates.length - 1];
  document.getElementById("datePicker").value = selectedDate;
  document.getElementById("tableDateFilter").min = dates[0];
  document.getElementById("tableDateFilter").max = dates[dates.length - 1];

  const regionSelect = document.getElementById("regionSelect");
  regionSelect.innerHTML = regions.map((r) => `<option value="${r}">${r}</option>`).join("");
  regionSelect.value = selectedRegion;

  document.getElementById("datePicker").addEventListener("change", (event) => {
    selectedDate = event.target.value;
    renderAll();
  });
  regionSelect.addEventListener("change", (event) => {
    selectedRegion = event.target.value;
    renderAll();
  });
  document.getElementById("riskFilter").addEventListener("change", renderTable);
  document.getElementById("tableDateFilter").addEventListener("change", renderTable);

  document.getElementById("modelStatus").textContent =
    `${metrics.selected_model} model · ${metrics.prediction_horizon_days}-day forecast`;
}

function initializeMap() {
  map = L.map("map", { zoomControl: true }).setView([40.35, 47.8], 7);
  const normal = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "OpenStreetMap",
  }).addTo(map);
  const satellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 18, attribution: "Esri World Imagery" },
  );
  L.control.layers({ "Normal map": normal, "Satellite layer": satellite }, null, { collapsed: false }).addTo(map);
}

function latestByRegion() {
  return forecast.filter((d) => d.date === selectedDate);
}

function rowForSelection() {
  return forecast.find((d) => d.region === selectedRegion && d.date === selectedDate)
    || forecast.find((d) => d.region === selectedRegion)
    || forecast[0];
}

function renderHeroStats() {
  const dayRows = latestByRegion();
  const highCount = dayRows.filter((d) => ["High", "Extreme"].includes(d.risk_level)).length;
  const avgProb = dayRows.reduce((sum, d) => sum + d.probability, 0) / dayRows.length;
  const maxRow = dayRows.reduce((a, b) => (a.probability > b.probability ? a : b), dayRows[0]);
  document.getElementById("heroStats").innerHTML = [
    ["Regions", new Set(forecast.map((d) => d.region)).size],
    ["High-risk areas", highCount],
    ["Average risk", fmtPct(avgProb)],
    ["Watch region", maxRow.region],
  ].map(([label, value]) => `<div class="stat-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function markerHtml(row, isSelected) {
  const size = isSelected ? 24 : 18;
  return `
    <div style="
      width:${size}px;height:${size}px;border-radius:50%;
      background:${riskColors[row.risk_level]};
      border:3px solid white;
      box-shadow:0 10px 24px rgba(15,42,67,.28);
    "></div>`;
}

function renderMap() {
  markers.forEach((marker) => marker.remove());
  markers = [];
  latestByRegion().forEach((row) => {
    const isSelected = row.region === selectedRegion;
    const circle = L.circle([row.Latitude, row.Longitude], {
      radius: 20000,
      color: riskColors[row.risk_level],
      weight: isSelected ? 3 : 1,
      fillColor: riskColors[row.risk_level],
      fillOpacity: isSelected ? 0.22 : 0.12,
    }).addTo(map);
    const marker = L.marker([row.Latitude, row.Longitude], {
      icon: L.divIcon({
        className: "",
        html: markerHtml(row, isSelected),
        iconSize: [26, 26],
        iconAnchor: [13, 13],
      }),
    }).addTo(map);
    marker.bindTooltip(`${row.region}: ${row.risk_level} (${fmtPct(row.probability)})`);
    const selectRegion = () => {
      selectedRegion = row.region;
      document.getElementById("regionSelect").value = row.region;
      renderAll();
    };
    marker.on("click", selectRegion);
    circle.on("click", selectRegion);
    marker.on("mouseover", () => marker.openTooltip());
    markers.push(circle, marker);
  });
}

function renderPanel() {
  const row = rowForSelection();
  document.getElementById("selectedRegion").textContent = row.region;
  document.getElementById("riskLevel").textContent = row.risk_level;
  document.getElementById("riskScore").textContent = fmtPct(row.probability);
  document.getElementById("riskBand").style.background = riskColors[row.risk_level];
  document.getElementById("panelTemp").textContent = `${fmtOne(row.temperature)} C`;
  document.getElementById("panelWind").textContent = `${fmtOne(row.wind)} km/h`;
  document.getElementById("panelHumidity").textContent = `${fmtOne(row.humidity)}%`;
  document.getElementById("panelConfidence").textContent = fmtPct(row.confidence);
  document.getElementById("panelSummary").textContent = row.climate_summary;
  document.getElementById("panelWarning").textContent = row.warning;

  const regionRows = forecast.filter((d) => d.region === selectedRegion);
  Plotly.react("trendChart", [{
    x: regionRows.map((d) => d.date),
    y: regionRows.map((d) => d.probability * 100),
    type: "scatter",
    mode: "lines",
    line: { color: "#376A8F", width: 3 },
    fill: "tozeroy",
    fillcolor: "rgba(91,143,185,.16)",
    hovertemplate: "%{x}<br>%{y:.1f}%<extra></extra>",
  }], {
    margin: { t: 8, r: 8, b: 36, l: 38 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    yaxis: { title: "", ticksuffix: "%", gridcolor: "#e3edf4" },
    xaxis: { title: "", gridcolor: "#eef4f8" },
  }, { displayModeBar: false, responsive: true });
}

function renderCalendar() {
  const rows = forecast.filter((d) => d.region === selectedRegion);
  document.getElementById("calendarGrid").innerHTML = rows.map((row) => `
    <div class="day-cell" style="background:${riskColors[row.risk_level]}" data-date="${row.date}">
      <small>${new Date(row.date).toLocaleDateString("en", { month: "short", day: "numeric" })}</small>
      <span>${row.risk_level}</span>
      <small>${fmtPct(row.probability)}</small>
    </div>
  `).join("");
  document.querySelectorAll(".day-cell").forEach((cell) => {
    cell.addEventListener("click", () => {
      selectedDate = cell.dataset.date;
      document.getElementById("datePicker").value = selectedDate;
      renderAll();
    });
  });
}

function renderCharts() {
  const regionRows = forecast.filter((d) => d.region === selectedRegion);
  Plotly.react("mainTrendChart", [{
    x: regionRows.map((d) => d.date),
    y: regionRows.map((d) => d.probability * 100),
    type: "bar",
    marker: { color: regionRows.map((d) => riskColors[d.risk_level]) },
    hovertemplate: "%{x}<br>Risk %{y:.1f}%<extra></extra>",
  }], {
    margin: { t: 10, r: 16, b: 44, l: 46 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    yaxis: { ticksuffix: "%", gridcolor: "#e3edf4" },
    xaxis: { gridcolor: "#eef4f8" },
  }, { displayModeBar: false, responsive: true });

  Plotly.react("weatherChart", [
    {
      x: regionRows.map((d) => d.date),
      y: regionRows.map((d) => d.temperature),
      name: "Temperature C",
      type: "scatter",
      mode: "lines",
      line: { color: "#376A8F", width: 3 },
    },
    {
      x: regionRows.map((d) => d.date),
      y: regionRows.map((d) => d.wind),
      name: "Wind km/h",
      type: "scatter",
      mode: "lines",
      line: { color: "#D96C3B", width: 3 },
    },
  ], {
    margin: { t: 10, r: 16, b: 44, l: 46 },
    legend: { orientation: "h", y: 1.15 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    yaxis: { gridcolor: "#e3edf4" },
    xaxis: { gridcolor: "#eef4f8" },
  }, { displayModeBar: false, responsive: true });
}

function renderTable() {
  const riskFilter = document.getElementById("riskFilter").value;
  const dateFilter = document.getElementById("tableDateFilter").value;
  let rows = forecast.filter((d) => d.region === selectedRegion);
  if (riskFilter !== "All") rows = rows.filter((d) => d.risk_level === riskFilter);
  if (dateFilter) rows = rows.filter((d) => d.date === dateFilter);
  document.getElementById("forecastTable").innerHTML = rows.map((row) => `
    <tr>
      <td>${row.date}</td>
      <td>${row.region}</td>
      <td><span class="risk-chip" style="background:${riskColors[row.risk_level]}">${row.risk_level}</span></td>
      <td>${fmtPct(row.probability)}</td>
      <td>${fmtOne(row.temperature)} C</td>
      <td>${fmtOne(row.wind)} km/h</td>
      <td>${fmtOne(row.humidity)}%</td>
    </tr>
  `).join("");
}

function renderAll() {
  renderHeroStats();
  renderMap();
  renderPanel();
  renderCalendar();
  renderCharts();
  renderTable();
}

loadData().catch((error) => {
  document.body.innerHTML = `<main class="section"><h2>Dashboard data could not be loaded</h2><p>${error.message}</p></main>`;
});
