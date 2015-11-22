/*global d3, $*/
/*jslint unparam:true */

/* HELPERS */

// REUSE: http://stackoverflow.com/questions/14167863/how-can-i-bring-a-circle-to-the-front-with-d3
d3.selection.prototype.moveToFront = function() {
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};

function color(d) { 
  var v = Math.floor(d.value[0]);
  return 'rgb(' + [v, v, v].join(',') + ')'; 
}

/* DISPLAY THE GOAL */

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
      .style('fill', color);

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
    selection
      .attr('x', d3.event.x - d3.select(this).attr('width') / 2)
      .attr('y', d3.event.y - d3.select(this).attr('height') / 2)
      .moveToFront();

    // Highlight nearby box it will replace
    var displaced = getDisplacedNeighbor(selection);
    d3.selectAll('#rank_bar rect').style('stroke-width', 0);
    displaced
      .style('stroke-width', 3)
      .style('stroke', 'rgb(166, 166, 255)');

  }).on('dragend', function() {
    
    // Rearrange the boxes based on where the current selection is dragged
    var selection = d3.select(this);
    var origX = selection.datum().origX;
    var displaced = getDisplacedNeighbor(selection);
    var newX = displaced.attr('x');
    console.log("Original X:" + origX);
    console.log("New X:" + newX);
    selection.attr('x', newX);
    displaced.attr('x', origX);

    var boxes = d3.selectAll('#rank_bar rect');
    boxes.each(function(d, i) {
      d.x = $(this).attr('x');
    });
    boxes.sort(function(a, b) {
      return a.x - b.x;
    });
    boxes
      .attr('x', function(d, i) { return i * 100; })
      .attr('y', 0)
      .datum(function(d, i) { 
        d.origX = d3.select(this).attr('x');
        d.rank = i;
        return d; 
      })
      .style('stroke-width', 0);

  });

/* LOAD THE EXAMPLES */

function makeData(vertices, extras) {
    var i;
    var data = [];
    for (i = 0; i < vertices.length; i++) {
        data.push({'rank': i, 'value': vertices[i], 'type': 'vertex'});
    }
    if (extras !== undefined) {
        for (i = 0; i < extras.length; i++) {
            data.push({
                'rank': vertices.length + i, 
                'value': extras[i].value,
                'type': extras[i].type
            });
        }
    }
    return data;
}

function loadExamples(vertices, extras) {

    var data = makeData(vertices, extras);

    d3.select('#rank_bar svg').remove();
    var rankSvg = d3.select('#rank_bar')
      .append('svg')
      .attr('width', 100 * data.length)
      .attr('height', 100);

    var boxes = rankSvg.selectAll('rect')
      .data(data)
      .enter()
        .append('rect')
          .attr('x', function(d, i) { return i * 100; } )
          .datum(function(d, i) { 
            d.origX = i * 100;
            return d; 
          }).attr('width', 100)
          .attr('height', 100)
          .datum(function(d) { 
            d.origX = d3.select(this).attr('x');
            return d; 
          })
          .style('fill', color)
          .call(drag);

    return boxes;
}

var boxes = loadExamples([[255], [172], [100], [32]]);

$('#upload_ranking_butt').click(function() {

    function getDataOfType(data, type) {
        var i;
        var typedData = [];
        for (i = 0; i < data.length; i++) {
            if (data[i].type === type) {
                typedData.push(data[i]);
            }
        }
        return typedData;
    }

    var allData = d3.selectAll('#rank_bar rect').data();
    var vData = getDataOfType(allData, 'vertex');

    function values(data) { 
        return data.map(function (e) { 
            return e.value; 
        });
    }
    function ranks(data) { 
        return data.map(function (e) { 
            return e.rank; 
        });
    }
    function rankOfType(data, type) {
        return ranks(getDataOfType(data, type))[0];
    }

    if (vData.length === allData.length) {
        $.get('/get_next', {
            'vertices': JSON.stringify(values(vData)),
            'vertex_ranks': ranks(vData),
        }, function(data) {
            var origData = values(vData);
            loadExamples(origData, [
                {'value': data.reflection, 'type': 'reflection'},
                {'value': data.expansion, 'type': 'expansion'},
                {'value': data.contraction, 'type': 'contraction'},
            ]);       
        });
    } else {
        $.get('/update_vertices', {
            'vertices': JSON.stringify(values(vData)),
            'vertex_ranks': ranks(vData),
            'reflection_rank': rankOfType(allData, 'reflection'),
            'expansion_rank': rankOfType(allData, 'expansion'),
            'contraction_rank': rankOfType(allData, 'contraction'),
        }, function(data) {
            loadExamples(data.vertices);
        });
    }

});
