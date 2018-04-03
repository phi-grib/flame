
// generates a random path for saving prediction results
function randomDir() {
    var text = "flame-";
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (var i = 0; i < 10; i++)
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}

function parseResults (results) {
    console.log('predict completed, parsing results')
    $("#data-body").text(results);

    var myjson = JSON.parse(results);

    var tbl_body = '<thead><tr><th>#mol</th><th>prediction</th></tr></thead>';
    var tbl_row;
    $.each(myjson, function() {
      tbl_row = "";

      $.each(this, function(k , v) {
        tbl_body += "<tr><td>"+(k+1)+"</td><td>"+v+"</td></tr>"; 
      })         

    })
    $("#data-table").html(tbl_body);   
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

    xhr.ontimeout = function () {
        console.log('WARNING! timed out. File can be incomplete');
        return false;
    };

    xhr.onload = function () {
        
        var version = $("#version option:selected").text();
        if (version=='dev') {
          version = '0';
        }

        // send job
        $.post("/predict", {"ifile"   : file.name,
                            "model"   : $("#myselect option:selected").text(),
                            "version" : version,
                            "temp_dir": temp_dir
                            })

        // show results
        .done(function(results) {
            parseResults (results);
        });
    }

    xhr.open('POST', '/upload', true); 
    xhr.timeout = 600000;
    xhr.setRequestHeader('X-Filename', file.name);
    xhr.setRequestHeader('Temp-Dir', temp_dir);

    console.log('sending', file.name, file);
    xhr.send(file);

    return true;
}