let droppedFiles = false;
let fileName = '';
const dropzone = document.querySelector('.dropzone');
const button = document.querySelector('.upload-btn');
let uploading = false;
const syncing = document.querySelector('.syncing');
const done = document.querySelector('.done');
const bar = document.querySelector('.bar');
let timeoutID;

['drag', 'dragstart', 'dragend', 'dragover', 'dragenter', 'dragleave', 'drop'].forEach(event => {
    dropzone.addEventListener(event, function(e) {
        e.preventDefault();
        e.stopPropagation();
    });
});

dropzone.addEventListener('dragover', function() {
    dropzone.classList.add('is-dragover');
});

['dragleave', 'dragend', 'drop'].forEach(event => {
    dropzone.addEventListener(event, function() {
        dropzone.classList.remove('is-dragover');
    });
});

dropzone.addEventListener('drop', function(e) {
    droppedFiles = e.dataTransfer.files;
    fileName = droppedFiles[0].name;
    document.querySelector('.filename').innerHTML = fileName;
    document.querySelector('.dropzone .upload').style.display = 'none';
});

button.addEventListener('click', function() {
    startUpload();
});

document.querySelector('input[type="file"]').addEventListener('change', function() {
    fileName = this.files[0].name;
    document.querySelector('.filename').innerHTML = fileName;
    document.querySelector('.dropzone .upload').style.display = 'none';
});

function startUpload() {
    if (!uploading && fileName !== '') {
        uploading = true;
        button.innerHTML = 'Uploading...';
        dropzone.style.display = 'none';
        syncing.classList.add('active');
        done.classList.add('active');
        bar.classList.add('active');
        timeoutID = window.setTimeout(showDone, 3200);
    }
}

function showDone() {
    button.innerHTML = 'Done';
}