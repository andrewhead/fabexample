/*globals $ */
/*jslint unparam:true */

var sliderConfigs = [
    {name: 'power', min: 1, max: 100, step: 0.1, suffix: '%'},
    {name: 'speed', min: 1, max: 100, step: 1, suffix: '%'},
    {name: 'ppi', min: 10, max: 1000, step: 10, suffix: ''},
];

function initSlider(config) {
    var slider = $('#' + config.name + '_slider');
    var label = $('#' + config.name + '_label');
    slider.slider({
        orientation: 'vertical',
        // range: 'min',
        min: config.min,
        max: config.max,
        value: config.max / 2,
        step: config.step,
        slide: function (event, ui) {
            label.val(ui.value + config.suffix);
        }
    });
    label.val(slider.slider('value') + config.suffix);
    label.change(function() {
        var value = $(this).val().replace(config.suffix, '');
        slider.slider('value', value);
        $(this).val(slider.slider('value') + config.suffix);
    });
}

var i;
for (i = 0; i < sliderConfigs.length; i++) {
    initSlider(sliderConfigs[i]);
}

$('#upload_ranking_butt').click(function() {
    $.get('/submit_job', {
        'values': [
            $('#power_slider').slider('value'),
            $('#speed_slider').slider('value'),
            $('#ppi_slider').slider('value'),
        ]
    });
});
