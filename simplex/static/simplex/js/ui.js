/*global d3, $*/
/*jslint unparam:true */

/* GLOBALS */

var MODE = '2D';
var iterationIndex = 1;
var exampleIndex = 1;

/* HELPERS */

// REUSE: http://stackoverflow.com/questions/14167863/how-can-i-bring-a-circle-to-the-front-with-d3
d3.selection.prototype.moveToFront = function() {

  return this.each(function(){
    this.parentNode.appendChild(this);
  });

};

function appearance(selection) {
    if (MODE === '1D') {
        selection.style('fill', function(d) {
            var v = Math.floor(d.value[0]);
            return 'rgb(' + [v, v, v].join(',') + ')';
        });
    } else {
        selection
          .style('fill', 'white')
          .style('stroke', 'black')
          .style('stroke-width', 2);
        d3.select('svg').selectAll('text')
          .data(selection.data())
          .enter()
            .append('text')
            .attr('transform', 'translate(10,25)')
            .style('stroke-width', 1)
            .style('font-size', 20)
            .style('stroke', 'black')
            .style('fill', 'white')
            .style('font-family', 'Helvetica')
            .text(function(d) { return d.index; });
    }
}

function move(selection, x, y) {

    selection.attr('x', x);
    if (y !== undefined) {
        selection.attr('y', y);
    }
    selection.moveToFront();

    function getMatchingText(box) {
        var text;
        d3.selectAll('#rank_panel text').each(function(d) {
          if (d.index === box.datum().index) {
              text = d3.select(this);   
          }
        });
        return text;
    }

    if (MODE !== '1D') {
        selection.each(function() {
          var box = d3.select(this);
          var text = getMatchingText(d3.select(this));
          text
            .attr('x', box.attr('x'))
            .attr('y', box.attr('y'))
            .moveToFront();
        });
    }
    

}

/* DISPLAY THE GOAL */

if (MODE === '1D') {

    d3.select('#simplex_exemplar_img').remove();

    var exemplarSvg = d3.select('#exemplar_cont')
      .append('svg')
      .attr('width', 200)
      .attr('height', 200);

    exemplarSvg.selectAll('rect')
      .data([{'value': [128]}])
      .enter()
        .append('rect')
          .attr('width', 200)
          .attr('height', 200)
          .call(appearance);
          // .style('fill', appearance);

}

/* MANIPULATE THE EXAMPLES */

function getDisplacedNeighbor(sel) {
    var displaced = d3.selectAll('.nothing');
    d3.selectAll('#rank_bar rect').each(function() {
      var neighbor = d3.select(this);
      if (neighbor.node() !== sel.node()) {
        if (Math.abs(neighbor.attr('x') - sel.attr('x')) < 50) {
           displaced = neighbor;
        }
      }
    });
    return displaced;
}

/* Drag and snap boxes within a line */
// REUSE: Based on drag behavior from
// stackoverflow.com/questions/19931307/d3-differentiate-between-click-and-drag-for-an-element-which-has-a-drag-behavior
var drag = d3.behavior.drag()
  .on('drag', function() {

    // Move the box
    var selection = d3.select(this);
    selection.call(move,
        d3.event.x - d3.select(this).attr('width') / 2,
        d3.event.y - d3.select(this).attr('height') / 2);

    // Highlight nearby box it will replace
    var displaced = getDisplacedNeighbor(selection);
    d3.selectAll('#rank_bar rect')
      .style('stroke', 'black')
      .style('stroke-width', 2);
    displaced
      .style('stroke-width', 3)
      .style('stroke', 'rgb(166, 166, 255)');

  }).on('dragend', function() {
    
    // Rearrange the boxes based on where the current selection is dragged
    var selection = d3.select(this);
    var displaced = getDisplacedNeighbor(selection);
    var newX = displaced.attr('x') - 10;
    selection.call(move, newX);

    var boxes = d3.selectAll('#rank_bar rect');
    boxes.each(function(d, i) {
      d.x = $(this).attr('x');
    });
    boxes.sort(function(a, b) {
      return a.x - b.x;
    });
    boxes
      .call(move, function(d, i) { return i * 100; }, 0)
      .datum(function(d, i) { 
        d.origX = d3.select(this).attr('x');
        d.rank = i;
        return d; 
      })
      .style('stroke', 'black')
      .style('stroke-width', 2);

  });

/* LOAD THE EXAMPLES */

function makeData(points) {

    var i;
    var data = [];

    function addExample(value, rank, type, index) {
        if (index === undefined) {
            index = exampleIndex;
            exampleIndex = exampleIndex + 1;
        }
        if (type === undefined) {
            type = "vertex";
        }
        data.push({
            'rank': rank,
            'index': index,
            'value': value,
            'type': type,
        });
    }

    for (i = 0; i < points.length; i++) {
        addExample(points[i].value, i, points[i].type, points[i].index);
    }

    return data;

}

function loadExamples(points) {

    var data = makeData(points);

    d3.select('#rank_bar svg').remove();
    var rankSvg = d3.select('#rank_bar')
      .append('svg')
      .attr('width', 100 * data.length)
      .attr('height', 100);

    var boxes = rankSvg.selectAll('rect')
      .data(data)
      .enter()
        .append('rect')
          .call(appearance)
          .attr('width', 100)
          .attr('height', 100)
          .datum(function(d) { 
            d.origX = d3.select(this).attr('x');
            return d; 
          })
          .call(move, function(d, i) { return i * 100; })
          .call(drag);

    iterationIndex = iterationIndex + 1;

    return boxes;
}

var boxes;

if (MODE === '1D') {
    boxes = loadExamples([
        {value: [255]},
        {value: [100]}, 
        {value: [32]}
    ]);
} else {
    /*
    boxes = loadExamples([
        {value: [100, 100, 1000]},
        {value: [10, 1, 10]}, 
        {value: [3, 100, 1000]},
    ]);
    */
    boxes = loadExamples([
        {value: [5, 5, 5]},
        {value: [3, 1, 1]}, 
        {value: [2, 5, 5]},
    ]);
}

$('#upload_ranking_butt').click(function() {

    var data = d3.selectAll('#rank_bar rect').data();
    $.get('/step', {
        'iteration': iterationIndex,
        'points': JSON.stringify(data),
    }, function(data) {
        loadExamples(data.points);
    });

});
