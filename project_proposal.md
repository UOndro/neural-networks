

# Odporúčací systém založený na LSTM neurónových sieťach

Viktória Markovičová, Ondrej Unger

## Motivácia
Odporúčacie systémy sa v súčasnosti tešia veľkej popularite. S nárastom informácií na internete vznikol problém s vyhľadávaním a vybraním si informácií relevantných pre nás. Práve tento problém nám majú odporúčacie systémy pomôcť riešiť. Ich hlavnou úlohou je odporučiť používateľovi taký produkt, ktorý bude preňho zaujímavý.
V zadaní budeme riešiť úlohu odporúčania produktu pomocou metódy next item prediction. Zo sekvencie hodnotení používateľa sa budeme snažiť predikovať ďalší film, ktorý si pozrie a ten mu odporučíme. Ďalšou úlohou, ktorú budeme riešiť v zadaní je zobrazenie jednotlivých filmov vo vektorovej podobe, tak aby filmy s podobnými vlastnosťami boli blízko seba. Následne v sekvencií používateľa zameníme filmy za ich číselnú reprezentáciu. Po zámene očakávame, že sa výsledné odporúčanie zlepší.
Pri riešení zadania budeme skúmať úspešnosť LSTM v oblasti odporúčaní. Taktiež budeme porovnávať použitie dát v textovej a vektorovej podobe a jeho vplyv na výsledné predikcie.
## Podobné práce
[Kolaboratívne filtrovanie s využitím rekurentných neurónových sietí](https://arxiv.org/pdf/1608.07400v2.pdf) - práca sa venuje skúmaniu kolaboratívneho filtrovania z pohľadu sekvenčnej predikcie. Na generovanie odporúčaní využíva LSTM neurónové siete, vďaka ktorým dosahuje lepšie výsledky, ako v prípade metód najbližších susedov alebo maticovej faktorizácie, a to najmä v oblasti krátkodobých odporúčaní.
[Dlhodobé a krátkodobé odporúčania s využitím rekurentných neurónových sietí](http://iridia.ulb.ac.be/~rdevooght/papers/UMAP__Long_and_short_term_with_RNN.pdf) - práca poskytuje vizualizáciu a porovnanie viacerých odporúčacích systémov. Taktiež skúma možné modifikácie rekurentných neurónových sietí za účelom prispôsobenia pre krátkodobé alebo dlhodobé odporúčania.
## Dataset
Na overenie úspešnosti nášho modelu sme si vybrali známy dataset Movielens. Dataset Movielens obsahuje hodnotenia filmov používateľmi, informácie o hodnotených filmoch a informácie o používateľoch. Movielens poskytuje viacero datasetov s rôznym množstvom dát. My sme si vybrali dataset o veľkosti 1 milióna. V datasete sa nachádza 6040 používateľov, pričom každý z nich má aspoň 20 hodnotení. Celkový počet filmov v datasete je 3952. Informácie o používateľoch v našom modeli nebudeme potrebovať.
### Akcie používateľa

|Názov|Typ|Popis|
|----------------|-------------------------------|------------------------|
|UserID|celé číslo|identifikačné číslo hodnotiaceho používateľa|
|MovieID|celé číslo|identifikačné číslo ohodnoteného filmu|
|Rating|celé číslo|používateľove hodnotenie filmu|
|Timestamp|desatinné číslo|čas vykonania hodnotenia|

### Informácie o filme
|Názov|Typ|Popis|
|----------------|-------------------------------|------------------------|
|MovieID|celé číslo|identifikačné číslo filmu|
|Title|text|názov filmu spolu s rokom vydania|
|Genres|text|žánre pod ktoré spadá film|

## Návrh
Ako prvý bod pri riešení nášho zadania namapujeme filmy do vektorového priestoru a to tak aby filmy, ktoré sa zvyknú pozerať spolu boli blízko seba. Tento problém budeme riešiť pomocou známej metódy v spracovaní prirodzeného jazyka word embedding. Ide o metódu, ktorá mapuje slová na číselnú reprezentáciu a to tak, aby sa slová s podobnými vlastnosťami nachádzali blízko seba. Naša metóda bude veľmi podobná, ale namiesto sekvencie slov budeme používať sekvenciu filmov v poradí, v akom boli hodnotené používateľmi. Ako ďalší krok vytvorené mapovanie spojíme spolu s obsahovými vlastnosťami filmov. V poslednom kroku, zoberieme sekvenciu používateľom hodnotených filmov, už namapovaných na číselnú reprezentáciu a využijeme neurónovú sieť LSTM (Long short-term memory) na predikciu ďalšieho filmu.
