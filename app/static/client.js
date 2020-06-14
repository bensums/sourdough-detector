var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function renderResponse(result) {
  var c = el("result-canvas");
  var ctx = c.getContext("2d");
  var img = el("image-picked");
  c.width = img.width;
  c.height = img.height;
  c.style.width = `${img.width}px`;
  c.style.height = `${img.height}px`;
  ctx.drawImage(img, 0, 0, img.width, img.height);

  var scaleX = img.width / img.naturalWidth;
  var scaleY = img.height / img.naturalHeight;

  result['predictions'].forEach(function(pred) {
    var bbox = pred['bbox'];
    var class_ = pred['class'];
    if (class_ != 'background') {
      ctx.beginPath();
      ctx.rect(bbox[1] * scaleX, bbox[0] * scaleY, bbox[3] * scaleX, bbox[2] * scaleY);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'black';
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'white';
      ctx.stroke();

      ctx.beginPath();
      ctx.font = "15px Arial";
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 3;
      var text = `${class_} ${Math.round(pred['score'] * 100) / 100}`
      ctx.strokeText(text, bbox[1] * scaleX, bbox[0] * scaleY + 10);
      ctx.fillStyle = 'white';
      ctx.fillText(text, bbox[1] * scaleX, bbox[0] * scaleY + 10);
    }
  });
}

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      el("result-label").innerHTML = `Result = ${response["result"]}`;
      renderResponse(response["result"]);
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}

