module simple (a, b, c, y);

    input a, b, c;
    output y;
    wire n1, n2;

    and g0(n1, a, b);
    or g1(n2, n1, c);
    not g2(y, n2);

endmodule