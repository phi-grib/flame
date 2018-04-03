
// generates a random path for saving prediction results
function randomDir() {
    var text = "flame-";
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (var i = 0; i < 10; i++)
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}

// AJAX upload function
function upload(file, temp_dir) {
    var xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', function(event) {          
        console.log('progess', file.name, event.loaded, event.total);
    });

    xhr.addEventListener('readystatechange', function(event) {
        console.log(
            'ready state', 
            file.name, 
            xhr.readyState, 
            xhr.readyState == 4 && xhr.status
        );
    });

    xhr.ontimeout = function (e) {
        console.log('WARNING! timed out. File can be incomplete');
        return false;
    };

    xhr.open('POST', '/upload', true);
    xhr.timeout = 600000;
    xhr.setRequestHeader('X-Filename', file.name);
    xhr.setRequestHeader('Temp-Dir', temp_dir);

    console.log('sending', file.name, file);
    xhr.send(file);

    return true;
}