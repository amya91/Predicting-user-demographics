//function to create a new checkbox object
function d3CheckBox (label,tag) {

    var size = 20,
        x = 0,
        y = 0,
        rx = 0,
        ry = 0,
        markStrokeWidth = 3,
        boxStrokeWidth = 3,
        checked = false,
        clickEvent,
        mark,
        percent = "0%";

    //function to provide attribute to checkbox     
    function checkBox (selection) {

        var g = selection.append("g"),
            box = g.append("rect")
            .attr("id",label)
            .attr("width", size)
            .attr("height", size)
            .attr("x", x)
            .attr("y", y)
            .attr("rx", rx)
            .attr("ry", ry)
            .style({
                "fill-opacity": 0,
                "stroke-width": boxStrokeWidth,
            })
            .style("stroke",function(){if(tag=="M"){return "steelblue";}
                                          else if (tag=="F"){return "#C76D6D";}  
            });

        selection.append('text')
            .attr("x", x-(size/2))
            .attr("y", y+(size*3/4))
            .attr("text-anchor", "end")
            .text(label)
            .style({
                "font-family": "Calibri",
                "font-size":"10px",
                "font-weight": "Bold"
            });        

        //Data to represent the check mark
        var coordinates = [
            {x: x + (size / 8), y: y + (size / 3)},
            {x: x + (size / 2.2), y: (y + size) - (size / 4)},
            {x: (x + size) - (size / 8), y: (y + (size / 10))}
        ];

        var line = d3.svg.line()
                .x(function(d){ return d.x; })
                .y(function(d){ return d.y; })
                .interpolate("basic");

        mark = g.append("path")
            .attr("d", line(coordinates))
            .style({
                "stroke-width" : markStrokeWidth,
                "fill" : "none",
                "opacity": (checked)? 1 : 0
            })
            .style("stroke",function(){if(tag=="M"){return "steelblue";}
                                          else if (tag=="F"){return "#C76D6D";}  
            });

        g.on("click", function () {
            checked = !checked;
            mark.style("opacity", (checked)? 1 : 0);

            if(clickEvent)
                clickEvent();

            d3.event.stopPropagation();
        });

        //function to calculate the percent of the selected class and visualize it
        checkBox.count = function(data,selection){
        if(label=="Male"||label=="Female"){
            f = [tag];
            if(data.length!=0){   
            percent = Number((data.filter(function(d,i){return f.includes(d.gender);}).length/data.length*100).toFixed(2))+"%" ;
            }
            else {
                percent = "0%"
            }
        }
        else {
            f = [tag+label];
            t = [tag]
            if(data.filter(function(d,i){return t.includes(d.gender);})!=0){
            percent = Number((data.filter(function(d,i){return f.includes(d.age);}).length/data.filter(function(d,i){return t.includes(d.gender);}).length*100).toFixed(2))+"%" ;
            }
            else {
                percent = "0%"
            }
        }

        g.select("#percent").remove();
        g.append('text')
            .attr("id","percent")
            .attr("x", x+(size*2))
            .attr("y", y+(size*3/4))
            .attr("text-anchor", "start")
            .text(percent)
            .style({
                "font-family": "Calibri",
                "font-size":"10px"
            });
        
    }

    }

    checkBox.unmark = function(){
        checked = false;
        mark.style("opacity", (checked)? 1 : 0)
    }

    checkBox.marked = function(){
        checked = true;
        mark.style("opacity", (checked)? 1 : 0)
    }

    checkBox.size = function (val) {
        size = val;
        return checkBox;
    }

    checkBox.x = function (val) {
        x = val;
        return checkBox;
    }

    checkBox.y = function (val) {
        y = val;
        return checkBox;
    }

    checkBox.rx = function (val) {
        rx = val;
        return checkBox;
    }

    checkBox.ry = function (val) {
        ry = val;
        return checkBox;
    }

    checkBox.label = function (val) {
        label = val;    
        return checkBox;
    }

    checkBox.markStrokeWidth = function (val) {
        markStrokeWidth = val;
        return checkBox;
    }

    checkBox.boxStrokeWidth = function (val) {
        boxStrokeWidth = val;
        return checkBox;
    }

    checkBox.checked = function (val) {

        if(val === undefined) {
            return checked;
        } else {
            checked = val;
            return checkBox;
        }
    }

    checkBox.clickEvent = function (val) {
        clickEvent = val;
        return checkBox;
    }

    return checkBox;
}