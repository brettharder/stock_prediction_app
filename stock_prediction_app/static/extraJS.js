function loadStock(e){
    e.preventDefault();
    console.log("Add this!");
    /*GET Name of stock to predict...
    USE AJAX call to send it to python...
    then train the model real fast on server...
    On success pass data back and plot prediction and graphics
    */
    
    /*
    $.ajax({
    type: "GET",
    data: {
        'XValue': $('#XValue').val(),
        'YValue':$('#YValue').val()
    },
    success: function(response){
        $.get("/?ReturnData=True", function(data){
            $('body').html(data);
        });
    }
    });
    */

}


TESTER = document.getElementById('tester');
	Plotly.newPlot( TESTER, [{
	x: [1, 2, 3, 4, 5],
	y: [1, 2, 4, 8, 16] }], {
	margin: { t: 0 } } );
