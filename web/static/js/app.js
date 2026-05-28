let currentJobId = null;
let currentScanId = null;

function updateSensitivityLabel(value) {
  var el = document.getElementById("sensitivityValue");
  if (el) el.textContent = parseFloat(value).toFixed(2);
}

// Auto-reload do browser quando arquivos mudam no servidor
(async function () {
  var initial = null;
  setInterval(async function () {
    try {
      var res = await fetch("/api/version");
      var data = await res.json();
      if (initial === null) {
        initial = data.version;
      } else if (data.version !== initial) {
        console.log("Mudanças detectadas, recarregando...");
        location.reload();
      }
    } catch (e) {}
  }, 3000);
})();

// Selection is handled by handlePhotoSelection (see below).

// ========== DISK USAGE ==========

var diskUsageState = {
  limitGB: parseFloat(localStorage.getItem("diskLimitGB")) || 2,
};

function _formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(0) + " MB";
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

async function refreshDiskUsage() {
  try {
    var res = await fetch("/api/disk-usage");
    var data = await res.json();
    var limitBytes = diskUsageState.limitGB * 1024 * 1024 * 1024;
    var pct = limitBytes > 0 ? Math.min(100, (data.total_bytes / limitBytes) * 100) : 0;

    var fill = document.getElementById("diskUsageFill");
    var text = document.getElementById("diskUsageText");
    if (!fill || !text) return;

    fill.style.width = pct + "%";
    fill.classList.remove("warn", "danger");
    if (pct >= 90) fill.classList.add("danger");
    else if (pct >= 70) fill.classList.add("warn");

    text.textContent = _formatSize(data.total_bytes) + " / " + diskUsageState.limitGB.toFixed(1) + " GB";
    text.title = "results: " + _formatSize(data.results_bytes) + " · uploads: " + _formatSize(data.uploads_bytes);
  } catch (e) {
    /* silent */
  }
}

function initDiskUsage() {
  var input = document.getElementById("diskLimitInput");
  if (input) {
    input.value = diskUsageState.limitGB;
    input.addEventListener("change", function () {
      var v = parseFloat(this.value);
      if (!isNaN(v) && v > 0) {
        diskUsageState.limitGB = v;
        localStorage.setItem("diskLimitGB", String(v));
        refreshDiskUsage();
      } else {
        this.value = diskUsageState.limitGB;
      }
    });
  }
  refreshDiskUsage();
  setInterval(refreshDiskUsage, 10000);
}

loadPeople();
loadVideos();
initDiskUsage();

// ========== TABS ==========

function switchTab(section, tab, btn) {
  var prefix = section === "people" ? "people-" : "videos-";
  var tabs = btn.parentElement.querySelectorAll(".tab");
  tabs.forEach(function (t) { t.classList.remove("active"); });
  btn.classList.add("active");

  var parent = btn.closest(".section");
  var contents = parent.querySelectorAll(".tab-content");
  contents.forEach(function (c) { c.classList.remove("active"); });

  var target = document.getElementById(prefix + tab);
  if (target) target.classList.add("active");
}

// ========== PEOPLE ==========

async function loadPeople() {
  const res = await fetch("/api/people");
  const people = await res.json();
  const grid = document.getElementById("peopleGrid");
  const empty = document.getElementById("peopleEmpty");

  grid.innerHTML = "";

  if (people.length === 0) {
    empty.style.display = "block";
    return;
  }

  empty.style.display = "none";

  people.forEach(function (p) {
    const card = document.createElement("div");
    card.className = "person-card";

    let thumbHtml = '<div class="person-thumb-placeholder"></div>';
    if (p.thumb) {
      thumbHtml = '<img class="person-thumb" src="/api/people/' +
        encodeURIComponent(p.name) + '/photo/' + encodeURIComponent(p.thumb) +
        '" alt="' + escapeHtml(p.name) + '">';
    }

    card.innerHTML =
      thumbHtml +
      '<div class="person-info">' +
      '<div class="name">' + escapeHtml(p.name) + '</div>' +
      '<div class="count">' + p.photo_count + ' foto(s)</div>' +
      '</div>' +
      '<button class="btn btn-danger" onclick="deletePerson(\'' +
      escapeHtml(p.name).replace(/'/g, "\\'") +
      '\')">Remover</button>';
    grid.appendChild(card);
  });
}

// ========== MANUAL FACE TAGGER ==========
// User uploads photos. Server detects faces in each photo and returns one crop
// per face — the grid shows one card per FACE (not per file), so group photos
// expand into multiple taggable cards. Tags are reusable: type a name once,
// then click it to stamp on any subsequent face.

var manualState = {
  detectId: null,
  files: [],     // [{ id, face_id, thumbUrl, sourcePhoto, tag }]
  tags: [],      // unique tag names in creation order
  activeTag: null,
};
var _manualFileIdCounter = 0;

function _renderManualPool() {
  var pool = document.getElementById("tagPoolChips");
  pool.innerHTML = "";
  manualState.tags.forEach(function (name) {
    var chip = document.createElement("span");
    chip.className = "tag-chip" + (manualState.activeTag === name ? " active" : "");
    chip.textContent = name;
    chip.onclick = function () {
      manualState.activeTag = (manualState.activeTag === name) ? null : name;
      _renderManualPool();
    };
    var x = document.createElement("button");
    x.className = "tag-chip-x";
    x.textContent = "×";
    x.title = "Remover tag";
    x.onclick = function (e) {
      e.stopPropagation();
      _removePoolTag(name);
    };
    chip.appendChild(x);
    pool.appendChild(chip);
  });
}

function _renderManualGrid() {
  var grid = document.getElementById("manualPhotoGrid");
  grid.innerHTML = "";
  manualState.files.forEach(function (f) {
    var card = document.createElement("div");
    card.className = "manual-photo-card" + (f.tag ? " tagged" : "");
    card.onclick = function () {
      if (manualState.activeTag) {
        f.tag = manualState.activeTag;
        _renderManualGrid();
        _renderManualStatus();
      }
    };

    var img = document.createElement("img");
    img.src = f.thumbUrl;
    img.className = "manual-photo-thumb";
    card.appendChild(img);

    if (f.sourcePhoto) {
      var src = document.createElement("div");
      src.className = "manual-photo-source";
      src.title = f.sourcePhoto;
      src.textContent = f.sourcePhoto.length > 22
        ? f.sourcePhoto.substring(0, 20) + "…"
        : f.sourcePhoto;
      card.appendChild(src);
    }

    var label = document.createElement("div");
    label.className = "manual-photo-tag";
    if (f.tag) {
      var chip = document.createElement("span");
      chip.className = "tag-chip small";
      chip.textContent = f.tag;
      var x = document.createElement("button");
      x.className = "tag-chip-x";
      x.textContent = "×";
      x.title = "Tirar tag desta foto";
      x.onclick = function (e) {
        e.stopPropagation();
        f.tag = null;
        _renderManualGrid();
        _renderManualStatus();
      };
      chip.appendChild(x);
      label.appendChild(chip);
    } else {
      label.textContent = "Sem nome";
      label.classList.add("untagged");
    }
    card.appendChild(label);
    grid.appendChild(card);
  });
}

function _renderManualStatus() {
  var status = document.getElementById("selectedFiles");
  var tagged = manualState.files.filter(function (f) { return f.tag; }).length;
  if (manualState.files.length === 0) {
    status.textContent = "";
  } else {
    status.textContent = manualState.files.length + " foto(s), " + tagged + " com tag";
  }
}

function _renderManualTagger() {
  _renderManualPool();
  _renderManualGrid();
  _renderManualStatus();
}

async function handlePhotoSelection(input) {
  var files = Array.from(input.files);
  if (files.length === 0) return;

  manualState.files = [];
  manualState.tags = [];
  manualState.activeTag = null;
  manualState.detectId = null;

  document.getElementById("manualTagger").style.display = "block";
  document.getElementById("selectedFiles").textContent =
    "Enviando " + files.length + " foto(s) e detectando rostos…";
  _renderManualTagger();

  var fd = new FormData();
  files.forEach(function (f) { fd.append("photos", f); });

  try {
    var res = await fetch("/api/detect-photo-faces", { method: "POST", body: fd });
    var data = await res.json();
    if (data.error) {
      alert(data.error);
      cancelManualUpload();
      input.value = "";
      return;
    }

    manualState.detectId = data.detect_id;
    manualState.files = (data.faces || []).map(function (f) {
      return {
        id: ++_manualFileIdCounter,
        face_id: f.face_id,
        thumbUrl: f.crop_url,
        sourcePhoto: f.source_photo,
        tag: null,
      };
    });

    var msg = manualState.files.length + " rosto(s) detectado(s) em " + data.total_photos + " foto(s)";
    if (data.photos_with_no_faces > 0) {
      msg += " · " + data.photos_with_no_faces + " sem rostos";
    }
    if (manualState.files.length === 0) {
      msg = "Nenhum rosto detectado. Tente fotos com rostos mais visíveis ou maiores.";
    }
    document.getElementById("selectedFiles").textContent = msg;
    _renderManualTagger();
  } catch (e) {
    alert("Erro ao enviar fotos: " + e.message);
    cancelManualUpload();
  } finally {
    input.value = "";
  }
}

function createTagFromInput() {
  var inp = document.getElementById("newTagInput");
  var name = inp.value.trim();
  if (!name) return;
  if (manualState.tags.indexOf(name) === -1) {
    manualState.tags.push(name);
  }
  manualState.activeTag = name;
  inp.value = "";
  _renderManualPool();
}

function _removePoolTag(name) {
  var inUse = manualState.files.filter(function (f) { return f.tag === name; }).length;
  if (inUse > 0) {
    if (!confirm('Remover "' + name + '"? ' + inUse + ' foto(s) ficarão sem tag.')) return;
  }
  manualState.tags = manualState.tags.filter(function (t) { return t !== name; });
  manualState.files.forEach(function (f) { if (f.tag === name) f.tag = null; });
  if (manualState.activeTag === name) manualState.activeTag = null;
  _renderManualTagger();
}

async function saveAllTaggedPhotos() {
  var tagged = manualState.files.filter(function (f) { return f.tag; });
  if (tagged.length === 0) {
    alert("Nenhum rosto com tag pra salvar.");
    return;
  }
  if (!manualState.detectId) {
    alert("Sessão de detecção não encontrada. Selecione as fotos novamente.");
    return;
  }

  var assignments = tagged.map(function (f) {
    return { face_id: f.face_id, name: f.tag };
  });

  try {
    var res = await fetch("/api/save-tagged-faces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        detect_id: manualState.detectId,
        assignments: assignments,
      }),
    });
    var data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }

    var saved = data.saved || {};
    var names = Object.keys(saved);
    document.getElementById("manualTagger").style.display = "none";
    document.getElementById("selectedFiles").textContent =
      (data.total || 0) + " rosto(s) salvo(s) em " + names.length + " pessoa(s).";
    manualState.files = [];
    manualState.tags = [];
    manualState.activeTag = null;
    manualState.detectId = null;
    loadPeople();
  } catch (e) {
    alert("Erro ao salvar: " + e.message);
  }
}

function cancelManualUpload() {
  manualState.files = [];
  manualState.tags = [];
  manualState.activeTag = null;
  manualState.detectId = null;
  document.getElementById("manualTagger").style.display = "none";
  document.getElementById("selectedFiles").textContent = "";
}

async function deletePerson(name) {
  if (!confirm('Remover "' + name + '" e todas as suas fotos?')) return;
  await fetch("/api/people/" + encodeURIComponent(name), { method: "DELETE" });
  loadPeople();
}

// ========== FOLDER SCAN ==========

async function scanFolderFiles(input) {
  if (input.files.length === 0) return;

  var folderName = input.files[0].webkitRelativePath.split("/")[0];

  var photoExts = ["jpg", "jpeg", "png", "bmp", "webp"];
  var videoExts = ["mp4", "mov", "avi", "mkv", "webm", "m4v"];

  var photos = [];
  var videos = [];
  for (var i = 0; i < input.files.length; i++) {
    var ext = input.files[i].name.split(".").pop().toLowerCase();
    if (photoExts.indexOf(ext) !== -1) photos.push(input.files[i]);
    else if (videoExts.indexOf(ext) !== -1) videos.push(input.files[i]);
  }

  if (photos.length === 0 && videos.length === 0) {
    alert("Nenhuma foto ou vídeo encontrado na pasta.");
    input.value = "";
    return;
  }

  var parts = [];
  if (photos.length > 0) parts.push(photos.length + " foto(s)");
  if (videos.length > 0) parts.push(videos.length + " vídeo(s)");
  var partsLabel = parts.join(" + ");

  var msg = 'Pasta selecionada: "' + folderName + '"\n\n' +
    partsLabel + " encontrado(s).\n\n";

  if (photos.length > 200) {
    msg += "Essa pasta tem bastante conteúdo! O escaneamento pode demorar um pouco.\n\n";
  }
  if (videos.length > 0) {
    msg += "Os vídeos serão amostrados a 2 fps para extrair rostos.\n\n";
  }

  msg += "Deseja continuar com o escaneamento?";

  if (!confirm(msg)) {
    input.value = "";
    document.getElementById("refFolderName").textContent = "";
    return;
  }

  document.getElementById("refFolderName").textContent = folderName + " (" + partsLabel + ")";

  var formData = new FormData();
  for (var i = 0; i < photos.length; i++) formData.append("photos", photos[i]);
  for (var i = 0; i < videos.length; i++) formData.append("videos", videos[i]);
  formData.append("cluster_tolerance", document.getElementById("globalTolerance").value);

  var totalCount = photos.length + videos.length;
  document.getElementById("scanProgress").innerHTML =
    '<div class="scan-progress-bar">' +
    '<div class="scan-progress-track"><div class="scan-progress-fill" id="scanFill"></div></div>' +
    '<div class="scan-progress-info">' +
    '<span id="scanStatusText">Enviando ' + totalCount + ' arquivo(s)...</span>' +
    '<span id="scanPercent">0%</span></div></div>';

  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/scan-folder");

  xhr.upload.onprogress = function (e) {
    if (e.lengthComputable) {
      var pct = Math.round((e.loaded / e.total) * 100);
      var fill = document.getElementById("scanFill");
      var text = document.getElementById("scanStatusText");
      var pctEl = document.getElementById("scanPercent");
      if (fill) fill.style.width = pct + "%";
      if (pctEl) pctEl.textContent = pct + "%";
      var mb = (e.loaded / (1024 * 1024)).toFixed(0);
      var totalMb = (e.total / (1024 * 1024)).toFixed(0);
      if (text) text.textContent = "Enviando: " + mb + " / " + totalMb + " MB";
    }
  };

  xhr.onload = function () {
    var data = JSON.parse(xhr.responseText);
    if (data.error) {
      document.getElementById("scanProgress").textContent = "";
      alert(data.error);
      return;
    }
    currentScanId = data.scan_id;
    document.getElementById("scanStatusText").textContent = "Detectando rostos...";
    var fill = document.getElementById("scanFill");
    fill.style.width = "0%";
    fill.classList.add("processing");
    document.getElementById("scanPercent").textContent = "0%";
    pollScan(data.scan_id);
  };

  xhr.onerror = function () {
    document.getElementById("scanProgress").textContent = "";
    alert("Erro ao enviar imagens.");
  };

  xhr.send(formData);
  input.value = "";
}

function pollScan(scanId) {
  var interval = setInterval(async function () {
    var res = await fetch("/api/scan-status/" + scanId);
    var data = await res.json();

    if (data.total > 0) {
      var pct = Math.round((data.processed / data.total) * 100);
      var fill = document.getElementById("scanFill");
      var text = document.getElementById("scanStatusText");
      var pctEl = document.getElementById("scanPercent");

      if (fill) fill.style.width = pct + "%";
      if (pctEl) pctEl.textContent = pct + "%";
      var phaseLabel = (data.phase === "extracting") ? "extraindo frames" : "itens";
      if (text) text.textContent = data.processed + "/" + data.total +
        " " + phaseLabel + " | " + data.faces_found + " rosto(s) encontrado(s)";
    }

    if (data.status === "done") {
      clearInterval(interval);
      document.getElementById("scanProgress").textContent =
        data.result.length + " pessoa(s) unica(s) detectada(s). Nomeie abaixo:";
      showClusters(scanId, data.result);
    } else if (data.status === "error") {
      clearInterval(interval);
      document.getElementById("scanProgress").textContent = "";
      alert(data.error || "Erro no escaneamento");
    }
  }, 1000);
}

var clusterPage = 0;
var clusterPageSize = 8;
var allClusterFaces = [];
var currentClusterScanId = null;
var skippedClusters = {};
var selectedClusters = {};

function showClusters(scanId, faces) {
  var section = document.getElementById("clusteringSection");
  section.style.display = "block";
  allClusterFaces = faces;
  currentClusterScanId = scanId;
  clusterPage = 0;
  savedClusterNames = {};
  skippedClusters = {};
  selectedClusters = {};
  renderClusterPage();
}

function renderClusterPage() {
  var grid = document.getElementById("clusterGrid");
  grid.innerHTML = "";

  var start = clusterPage * clusterPageSize;
  var end = Math.min(start + clusterPageSize, allClusterFaces.length);
  var pageFaces = allClusterFaces.slice(start, end);
  var totalPages = Math.ceil(allClusterFaces.length / clusterPageSize);

  pageFaces.forEach(function (face) {
    var card = document.createElement("div");
    card.className = "cluster-card";
    if (skippedClusters[face.id]) card.classList.add("skipped");
    if (selectedClusters[face.id]) card.classList.add("selected");

    var suggestedName = face.suggested_name || savedClusterNames[face.id] || "";
    var matchTag = face.suggested_name
      ? '<span class="match-tag">Ja cadastrado(a)</span>'
      : '';

    card.innerHTML =
      '<button class="cluster-skip-btn" onclick="event.stopPropagation(); toggleSkip(' + face.id + ')" title="Pular este rosto">&times;</button>' +
      '<div class="cluster-select-overlay" onclick="toggleSelect(' + face.id + ')">' +
      '<img class="cluster-thumb" src="/api/scan-thumbs/' + currentClusterScanId + '/' + face.thumb + '">' +
      '<div class="cluster-check"></div>' +
      '</div>' +
      '<div class="cluster-info">' +
      matchTag +
      '<span class="count">' + face.photo_count + ' foto(s)</span>' +
      '<input type="text" class="cluster-name-input" data-id="' + face.id +
      '" data-original="' + escapeHtml(suggestedName) + '"' +
      ' onblur="confirmNameChange(this)"' +
      ' placeholder="Nome desta pessoa" value="' + escapeHtml(suggestedName) + '"' +
      (skippedClusters[face.id] ? ' disabled' : '') + '>' +
      '</div>';
    grid.appendChild(card);
  });

  renderClusterNav(end, totalPages);
  renderMergeBar();
}

function renderClusterNav(end, totalPages) {
  var nav = document.getElementById("clusterNav");
  if (!nav) {
    nav = document.createElement("div");
    nav.id = "clusterNav";
    nav.className = "cluster-nav";
    var confirmBtn = document.querySelector("#clusteringSection .btn-primary");
    confirmBtn.parentNode.insertBefore(nav, confirmBtn);
  }

  nav.innerHTML =
    '<span class="cluster-page-info">Pagina ' + (clusterPage + 1) + ' de ' + totalPages +
    ' (' + allClusterFaces.length + ' rostos)</span>' +
    '<div class="cluster-nav-btns">' +
    (clusterPage > 0
      ? '<button class="btn btn-secondary" onclick="prevClusterPage()">Anterior</button>'
      : '') +
    (end < allClusterFaces.length
      ? '<button class="btn btn-primary" onclick="nextClusterPage()">Proxima</button>'
      : '') +
    '</div>';
}

function renderMergeBar() {
  var bar = document.getElementById("mergeBar");
  var count = Object.keys(selectedClusters).length;

  if (count < 2) {
    if (bar) bar.remove();
    return;
  }

  if (!bar) {
    bar = document.createElement("div");
    bar.id = "mergeBar";
    bar.className = "merge-bar";
    document.body.appendChild(bar);
  }

  bar.innerHTML =
    '<span>' + count + ' rostos selecionados</span>' +
    '<input type="text" id="mergeNameInput" placeholder="Nome desta pessoa (ex: Pai da Noiva)">' +
    '<button class="btn btn-primary" onclick="mergeSelected()">Combinar</button>' +
    '<button class="btn btn-secondary" onclick="clearSelection()">Cancelar</button>';
}

function toggleSkip(id) {
  saveCurrentPageNames();
  if (skippedClusters[id]) {
    delete skippedClusters[id];
  } else {
    skippedClusters[id] = true;
    delete selectedClusters[id];
  }
  renderClusterPage();
  restorePageNames();
}

function toggleSelect(id) {
  if (skippedClusters[id]) return;
  saveCurrentPageNames();
  if (selectedClusters[id]) {
    delete selectedClusters[id];
  } else {
    selectedClusters[id] = true;
  }
  renderClusterPage();
  restorePageNames();
}

function clearSelection() {
  selectedClusters = {};
  renderClusterPage();
}

function mergeSelected() {
  var nameInput = document.getElementById("mergeNameInput");
  var name = nameInput.value.trim();
  if (!name) {
    alert("Digite um nome pra combinar.");
    return;
  }

  Object.keys(selectedClusters).forEach(function (id) {
    savedClusterNames[parseInt(id)] = name;
  });

  selectedClusters = {};
  renderClusterPage();
  restorePageNames();
}

function nextClusterPage() {
  saveCurrentPageNames();
  clusterPage++;
  renderClusterPage();
  restorePageNames();
  document.getElementById("clusteringSection").scrollIntoView({ behavior: "smooth" });
}

function prevClusterPage() {
  saveCurrentPageNames();
  clusterPage--;
  renderClusterPage();
  restorePageNames();
  document.getElementById("clusteringSection").scrollIntoView({ behavior: "smooth" });
}

function restorePageNames() {
  var inputs = document.querySelectorAll(".cluster-name-input");
  inputs.forEach(function (input) {
    var id = parseInt(input.dataset.id);
    if (savedClusterNames[id]) {
      input.value = savedClusterNames[id];
      input.dataset.original = savedClusterNames[id];
    }
  });
}

function confirmNameChange(input) {
  var original = input.dataset.original || "";
  var current = input.value.trim();
  if (!original || current === original || current === "") return;

  var msg = 'Mudar de "' + original + '" para "' + current + '"?';
  if (!confirm(msg)) {
    input.value = original;
  } else {
    input.dataset.original = current;
    var id = parseInt(input.dataset.id);
    savedClusterNames[id] = current;
  }
}

var savedClusterNames = {};

function saveCurrentPageNames() {
  var inputs = document.querySelectorAll(".cluster-name-input");
  inputs.forEach(function (input) {
    var id = parseInt(input.dataset.id);
    if (skippedClusters[id]) {
      delete savedClusterNames[id];
      return;
    }
    var name = input.value.trim();
    if (name) {
      savedClusterNames[id] = name;
    } else {
      delete savedClusterNames[id];
    }
  });
}

async function confirmClusters() {
  if (!currentScanId) return;

  saveCurrentPageNames();

  var assignments = [];
  Object.keys(savedClusterNames).forEach(function (id) {
    assignments.push({ id: parseInt(id), name: savedClusterNames[id] });
  });

  if (assignments.length === 0) {
    alert("Nomeie pelo menos uma pessoa.");
    return;
  }

  const res = await fetch("/api/confirm-people", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scan_id: currentScanId, assignments: assignments }),
  });
  const data = await res.json();

  if (data.error) {
    alert(data.error);
    return;
  }

  document.getElementById("clusteringSection").style.display = "none";
  document.getElementById("scanProgress").textContent =
    data.saved + " foto(s) salva(s) como referência.";
  currentScanId = null;
  loadPeople();
}

// ========== REF VIDEO ==========

async function scanRefVideo(input) {
  if (input.files.length === 0) return;

  var file = input.files[0];
  var sizeMB = (file.size / (1024 * 1024)).toFixed(0);
  document.getElementById("refVideoName").textContent = file.name + " (" + sizeMB + " MB)";

  var formData = new FormData();
  formData.append("video", file);
  formData.append("fps", document.getElementById("refVideoFps").value);
  formData.append("cluster_tolerance", document.getElementById("globalTolerance").value);

  var start = document.getElementById("refVideoStart").value.trim();
  var end = document.getElementById("refVideoEnd").value.trim();
  if (start) formData.append("start", start);
  if (end) formData.append("end", end);

  document.getElementById("refVideoProgress").innerHTML =
    '<div class="scan-progress-bar">' +
    '<div class="scan-progress-track"><div class="scan-progress-fill" id="refVidFill"></div></div>' +
    '<div class="scan-progress-info">' +
    '<span id="refVidStatusText">Enviando video...</span>' +
    '<span id="refVidPercent">0%</span></div></div>';

  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/scan-ref-video");

  xhr.upload.onprogress = function (e) {
    if (e.lengthComputable) {
      var pct = Math.round((e.loaded / e.total) * 100);
      var fill = document.getElementById("refVidFill");
      var text = document.getElementById("refVidStatusText");
      var pctEl = document.getElementById("refVidPercent");
      if (fill) fill.style.width = pct + "%";
      if (pctEl) pctEl.textContent = pct + "%";
      var mb = (e.loaded / (1024 * 1024)).toFixed(0);
      var totalMb = (e.total / (1024 * 1024)).toFixed(0);
      if (text) text.textContent = "Enviando: " + mb + " / " + totalMb + " MB";
    }
  };

  xhr.onload = function () {
    var data = JSON.parse(xhr.responseText);
    if (data.error) {
      document.getElementById("refVideoProgress").textContent = "";
      alert(data.error);
      return;
    }
    currentScanId = data.scan_id;
    var fill = document.getElementById("refVidFill");
    fill.style.width = "0%";
    fill.classList.add("processing");
    document.getElementById("refVidStatusText").textContent = "Extraindo frames...";
    document.getElementById("refVidPercent").textContent = "0%";
    pollRefVideoScan(data.scan_id);
  };

  xhr.onerror = function () {
    document.getElementById("refVideoProgress").textContent = "";
    alert("Erro ao enviar video.");
  };

  xhr.send(formData);
  input.value = "";
}

function pollRefVideoScan(scanId) {
  var interval = setInterval(async function () {
    var res = await fetch("/api/scan-status/" + scanId);
    var data = await res.json();

    var fill = document.getElementById("refVidFill");
    var text = document.getElementById("refVidStatusText");
    var pctEl = document.getElementById("refVidPercent");

    if (data.phase === "extracting") {
      if (text) text.textContent = "Extraindo frames do video...";
    } else if (data.total > 0) {
      var pct = Math.round((data.processed / data.total) * 100);
      if (fill) fill.style.width = pct + "%";
      if (pctEl) pctEl.textContent = pct + "%";
      if (text) text.textContent = data.processed + "/" + data.total +
        " frames | " + data.faces_found + " rosto(s)";
    }

    if (data.status === "done") {
      clearInterval(interval);
      document.getElementById("refVideoProgress").textContent =
        data.result.length + " pessoa(s) unica(s) detectada(s). Nomeie abaixo:";
      showClusters(scanId, data.result);
    } else if (data.status === "error") {
      clearInterval(interval);
      document.getElementById("refVideoProgress").textContent = "";
      alert(data.error || "Erro no escaneamento");
    }
  }, 1000);
}

// ========== VIDEOS ==========

async function loadVideos() {
  const res = await fetch("/api/videos");
  const videos = await res.json();
  const grid = document.getElementById("videosGrid");
  const empty = document.getElementById("videosEmpty");

  grid.innerHTML = "";

  if (videos.length === 0) {
    empty.style.display = "block";
    return;
  }

  empty.style.display = "none";

  videos.forEach(function (v) {
    const card = document.createElement("div");
    card.className = "video-card";
    var iconHtml = v.type === "photo"
      ? '<div class="video-icon" title="Foto">&#128247;</div>'
      : '<div class="video-icon" title="Vídeo">&#9654;</div>';
    var typeLabel = v.type === "photo" ? "Foto" : "Vídeo";
    card.innerHTML =
      iconHtml +
      '<div class="person-info">' +
      '<div class="name">' + escapeHtml(v.filename) + '</div>' +
      '<div class="count">' + typeLabel + ' · ' + v.size_mb + ' MB</div>' +
      '</div>' +
      '<button class="btn btn-danger" onclick="deleteVideo(\'' +
      escapeHtml(v.filename).replace(/'/g, "\\'") +
      '\')">Remover</button>';
    grid.appendChild(card);
  });
}

async function uploadVideos(input) {
  if (input.files.length === 0) return;

  const area = document.getElementById("videoArea");
  const label = document.getElementById("videoLabel");
  label.innerHTML = '<span class="filename">Enviando ' + input.files.length + ' arquivo(s)...</span>';
  area.classList.add("has-file");

  const formData = new FormData();
  for (let i = 0; i < input.files.length; i++) {
    formData.append("videos", input.files[i]);
  }

  const res = await fetch("/api/videos", { method: "POST", body: formData });
  const data = await res.json();

  label.textContent = "Clique para selecionar vídeos ou fotos (pode selecionar vários)";
  area.classList.remove("has-file");
  input.value = "";

  if (data.error) {
    alert(data.error);
    return;
  }

  loadVideos();
}

async function deleteVideo(filename) {
  if (!confirm("Remover este vídeo?")) return;
  await fetch("/api/videos/" + encodeURIComponent(filename), { method: "DELETE" });
  loadVideos();
}

async function loadVideoFolderFiles(input) {
  if (input.files.length === 0) return;

  var folderName = input.files[0].webkitRelativePath.split("/")[0];
  var videoExts = ["mp4", "mov", "avi", "mkv", "webm", "m4v"];
  var photoExts = ["jpg", "jpeg", "png", "bmp", "webp"];

  var videoCount = 0, photoCount = 0;
  var totalSize = 0;
  for (var i = 0; i < input.files.length; i++) {
    var ext = input.files[i].name.split(".").pop().toLowerCase();
    if (videoExts.indexOf(ext) !== -1) {
      videoCount++;
      totalSize += input.files[i].size;
    } else if (photoExts.indexOf(ext) !== -1) {
      photoCount++;
      totalSize += input.files[i].size;
    }
  }

  if (videoCount === 0 && photoCount === 0) {
    alert("Nenhum vídeo ou foto encontrado na pasta.");
    input.value = "";
    return;
  }

  var sizeMB = (totalSize / (1024 * 1024)).toFixed(0);
  var parts = [];
  if (videoCount > 0) parts.push(videoCount + " vídeo(s)");
  if (photoCount > 0) parts.push(photoCount + " foto(s)");
  var msg = 'Pasta selecionada: "' + folderName + '"\n\n' +
    parts.join(" + ") + " encontrado(s) (" + sizeMB + " MB total).\n\n" +
    "Deseja carregar?";

  if (!confirm(msg)) {
    input.value = "";
    document.getElementById("videoFolderName").textContent = "";
    return;
  }

  document.getElementById("videoFolderName").textContent = folderName;

  var formData = new FormData();
  for (var i = 0; i < input.files.length; i++) {
    var ext = input.files[i].name.split(".").pop().toLowerCase();
    if (videoExts.indexOf(ext) !== -1 || photoExts.indexOf(ext) !== -1) {
      formData.append("videos", input.files[i]);
    }
  }

  var totalCount = videoCount + photoCount;
  document.getElementById("videoFolderProgress").textContent =
    "Enviando " + totalCount + " arquivo(s)...";

  var res = await fetch("/api/load-videos-folder", { method: "POST", body: formData });
  var data = await res.json();

  if (data.error) {
    document.getElementById("videoFolderProgress").textContent = "";
    alert(data.error);
    return;
  }

  document.getElementById("videoFolderProgress").textContent =
    data.count + " arquivo(s) carregado(s).";
  input.value = "";
  loadVideos();
}

// ========== PROCESSING ==========

async function startProcessing() {
  const res1 = await fetch("/api/videos");
  const videos = await res1.json();

  if (videos.length === 0) {
    alert("Suba pelo menos um vídeo antes de processar.");
    return;
  }

  const videoFilenames = videos.map(function (v) { return v.filename; });

  const payload = {
    videos: videoFilenames,
    fps: parseFloat(document.getElementById("fpsInput").value),
    tolerance: parseFloat(document.getElementById("toleranceInput").value),
    start: document.getElementById("startInput").value.trim() || null,
    end: document.getElementById("endInput").value.trim() || null,
  };

  const btn = document.getElementById("processBtn");
  btn.disabled = true;
  btn.textContent = "Processando...";

  const progressBar = document.getElementById("progressBar");
  progressBar.classList.add("active");

  document.getElementById("resultsSection").classList.remove("visible");

  const res = await fetch("/api/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  if (data.error) {
    alert(data.error);
    btn.disabled = false;
    btn.textContent = "Processar Vídeos";
    progressBar.classList.remove("active");
    return;
  }

  currentJobId = data.job_id;
  pollJob(data.job_id);
}

function pollJob(jobId) {
  window.__lastSeenMatchIndex = 0;
  const liveBox = document.getElementById("liveMatches");
  if (liveBox) liveBox.innerHTML = "";

  const interval = setInterval(async function () {
    const res = await fetch("/api/jobs/" + jobId);
    const job = await res.json();

    var progressMsg = job.progress;
    if (job.partial_matches > 0 && job.status === "processing") {
      progressMsg += " | " + job.partial_matches + " match(es) ate agora";
    }
    document.getElementById("progressText").textContent = progressMsg;

    // Frame counter (per-video)
    const frameCounter = document.getElementById("frameCounter");
    if (frameCounter) {
      if (job.frames_total && job.status === "processing") {
        frameCounter.style.display = "block";
        frameCounter.textContent = "Frame " + job.frames_done + " / " + job.frames_total;
      } else {
        frameCounter.style.display = "none";
      }
    }

    // Incremental live matches
    const liveMatchBox = document.getElementById("liveMatches");
    if (liveMatchBox && Array.isArray(job.live_matches)) {
      if (typeof window.__lastSeenMatchIndex === "undefined") {
        window.__lastSeenMatchIndex = 0;
      }
      const slice = job.live_matches.slice(window.__lastSeenMatchIndex);
      for (const m of slice) {
        const row = document.createElement("div");
        row.className = "live-match-row";
        row.style.cssText = "padding: 4px 0; font-size: 13px; color: #ccc;";
        var parts = [];
        if (m.video) parts.push(m.video);
        if (m.timestamp) parts.push(m.timestamp);
        if (m.frame && m.frame !== m.video) parts.push(m.frame);
        row.textContent = parts.join(" · ");
        liveMatchBox.appendChild(row);
      }
      window.__lastSeenMatchIndex = job.live_matches.length;
    }

    if (job.status === "done") {
      clearInterval(interval);
      showResults(job, jobId);
      resetProcessButton();
      refreshDiskUsage();
    } else if (job.status === "error") {
      clearInterval(interval);
      alert("Erro: " + job.progress);
      resetProcessButton();
      refreshDiskUsage();
    }
  }, 2000);
}

function resetProcessButton() {
  const btn = document.getElementById("processBtn");
  btn.disabled = false;
  btn.textContent = "Processar Vídeos";
  document.getElementById("progressBar").classList.remove("active");
}

// ========== RESULTS ==========

function showResults(job, jobId) {
  const section = document.getElementById("resultsSection");
  const container = document.getElementById("resultsContainer");
  section.classList.add("visible");

  if (!job.results || Object.keys(job.results).length === 0) {
    container.innerHTML = '<div class="no-results">Nenhuma pessoa identificada nos frames.</div>';
    return;
  }

  let html = "";
  const names = Object.keys(job.results).sort();

  names.forEach(function (name) {
    const data = job.results[name];
    const bestMatches = data.best_matches || [];
    const videos = data.videos || [];

    html +=
      '<div class="result-person">' +
      '<div class="result-header">' +
      '<span class="result-name">' + escapeHtml(name) + "</span>" +
      '<span class="result-count">' + data.total_appearances + " aparicao(oes)</span>" +
      "</div>" +
      '<div class="result-confidence">Confianca media: ' +
      (data.avg_confidence * 100).toFixed(1) + "%";

    if (videos.length > 1) {
      html += " | Em " + videos.length + " videos";
    }

    html += "</div>";

    if (bestMatches.length > 0) {
      html += '<div class="match-grid">';
      bestMatches.forEach(function (m) {
        var imgSrc = m.match_image
          ? "/api/matches/" + jobId + "/" + m.match_image
          : "";

        html += '<div class="match-card">';
        if (imgSrc) {
          html += '<img class="match-img" src="' + imgSrc + '" alt="match">';
        }
        html +=
          '<div class="match-info">' +
          '<span class="match-ts">' + m.timestamp + "</span>" +
          '<span class="match-conf">' + (m.confidence * 100).toFixed(0) + "%</span>" +
          "</div>";
        if (m.video) {
          html += '<div class="match-video">' + escapeHtml(m.video) + "</div>";
        }
        html += "</div>";
      });
      html += "</div>";
    }

    var timestamps = data.timestamps || [];
    if (timestamps.length > 6) {
      html += '<div class="timestamps-list">';
      var remaining = timestamps.slice(6, 36);
      remaining.forEach(function (ts) {
        html += '<span class="timestamp-tag">' + ts + "</span>";
      });
      if (timestamps.length > 36) {
        html += '<span class="timestamp-tag">+' + (timestamps.length - 36) + " mais</span>";
      }
      html += "</div>";
    }

    html += "</div>";
  });

  container.innerHTML = html;
}

function downloadResults() {
  if (currentJobId) {
    window.location.href = "/api/jobs/" + currentJobId + "/download";
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function cleanupTemp() {
  if (!confirm("Remover todos os arquivos temporários (frames extraídos, scans antigos)?")) return;
  var res = await fetch("/api/cleanup", { method: "POST" });
  var data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  var resEl = document.getElementById("cleanupResult");
  if (resEl) resEl.textContent = data.freed_mb + " MB liberado(s)";
  refreshDiskUsage();
}
