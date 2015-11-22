/*globals $ */
/*jslint unparam:true */

var sliderConfigs = [
    {name: 'power', min: 1, max: 100, step: 0.1, suffix: '%'},
    {name: 'speed', min: 1, max: 100, step: 0.1, suffix: '%'},
    {name: 'ppi', min: 10, max: 1000, step: 10, suffix: ''},
];

function initSlider(config) {
    $('#' + config.name + '_slider').slider({
        orientation: 'vertical',
        // range: 'min',
        min: config.min,
        max: config.max,
        value: config.max / 2,
        step: config.step,
        slide: function (event, ui) {
            $('#' + config.name + '_label').val(ui.value + config.suffix);
        }
    });
    $('#' + config.name + '_label').val(
        $('#' + config.name + '_slider').slider('value') + config.suffix);
}

var i;
for (i = 0; i < sliderConfigs.length; i++) {
    initSlider(sliderConfigs[i]);
}
