-- kpi_overall
SELECT COUNT(DISTINCT Order_Id)                          AS orders,
       ROUND(100.0*AVG(Late_delivery_risk),1)            AS late_pct,
       ROUND(AVG(Days_for_shipping_real),2)              AS avg_actual_days,
       ROUND(AVG(Days_for_shipment_scheduled),2)         AS avg_promised_days,
       ROUND(SUM(Sales)/1e6,2)                           AS sales_m
FROM orders;

-- late_by_shipping_mode
SELECT Shipping_Mode,
       COUNT(*)                                          AS order_lines,
       ROUND(100.0*AVG(Late_delivery_risk),1)            AS late_pct,
       ROUND(AVG(Days_for_shipment_scheduled),1)         AS promised_days,
       ROUND(AVG(Days_for_shipping_real),1)              AS actual_days,
       ROUND(AVG(CASE WHEN delay_days>0 THEN delay_days END),2) AS avg_delay_when_late
FROM orders GROUP BY Shipping_Mode ORDER BY late_pct DESC;

-- late_by_market
SELECT Market, COUNT(*) AS order_lines, ROUND(100.0*AVG(Late_delivery_risk),1) AS late_pct,
       ROUND(SUM(Sales)/1e6,2) AS sales_m
FROM orders GROUP BY Market ORDER BY late_pct DESC;

-- worst_regions
SELECT Order_Region, COUNT(*) AS order_lines, ROUND(100.0*AVG(Late_delivery_risk),1) AS late_pct,
       RANK() OVER (ORDER BY AVG(Late_delivery_risk) DESC) AS late_rank
FROM orders GROUP BY Order_Region HAVING COUNT(*) > 1000 ORDER BY late_rank LIMIT 5;

-- profit_late_vs_ontime
SELECT CASE Late_delivery_risk WHEN 1 THEN 'Late' ELSE 'On time / early' END AS delivery,
       COUNT(*) AS order_lines, ROUND(AVG(Order_Profit_Per_Order),2) AS avg_profit_per_line
FROM orders GROUP BY 1;