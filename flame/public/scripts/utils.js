//     Description    Flame JavaScript utils
//
//     Authors:       Manuel Pastor (manuel.pastor@upf.edu)
// 
//     Copyright 2018 Manuel Pastor
// 
//     This file is part of Flame
// 
//     Flame is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation version 3.
// 
//     Flame is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
// 
//     You should have received a copy of the GNU General Public License
//     along with Flame. If not, see <http://www.gnu.org/licenses/>.

// generates a random path for saving prediction results
function randomDir() {
    var text = "flame-";
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (var i = 0; i < 10; i++)
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}

// AJAX upload function
function upload(file, temp_dir, postPredict) {
    var xhr = new XMLHttpRequest();

    // xhr.upload.addEventListener('progress', function(event) {          
    //     console.log('progess', file.name, event.loaded, event.total);
    // });

    // xhr.addEventListener('readystatechange', function(event) {
    //     console.log(
    //         'ready state', 
    //         file.name, 
    //         xhr.readyState, 
    //         xhr.readyState == 4 && xhr.status
    //     );
    // });

    xhr.ontimeout = function () {
        console.log('WARNING! timed out. File can be incomplete');
        return false;
    };

    xhr.onerror = function () {
        console.log('ERROR! file not uploaded');
        return false;
    }

    xhr.onload = function () {
        postPredict(temp_dir, file.name);
    };


    xhr.open('POST', '/upload', true); 
    xhr.timeout = 600000;

    xhr.setRequestHeader('X-Filename', file.name);
    xhr.setRequestHeader('Temp-Dir', temp_dir);

    xhr.send(file);

    return true;
}