# Anclas, no párrafos: extraccion de textos apalancados en LLMs de la fuente y no del modelo.

## El problema

Cuando hay que sacar información etiquetada de un documento —y esa información tiene que ser de confianza— lo habitual es poner un LLM de intermediario. El modelo lee un trozo del documento y devuelve el texto ya clasificado. La salida se parece a esto:

```json
{
  "items": [
    {
      "categoria": "definicion",
      "etiqueta": "160.103",
      "texto": "Breach means the acquisition, access, use, or disclosure of protected health information in a manner not permitted under subpart E of this part which compromises the security or privacy of the protected health information."
    },
    {
      "categoria": "definicion",
      "etiqueta": "160.103",
      "texto": "Business associate means a person who..."
    }
  ]
}
```

El modelo no solo decide la categoría. También **reescribe o copia el cuerpo entero** de cada unidad. Para que eso funcione hay que aceptar tres restricciones:

1. **Ventanas de contexto cortas.** La salida tiene que caber el texto completo más sus etiquetas. Si la unidad es larga, la ventana de entrada se achica para dejar sitio a la respuesta.
2. **Ventanas solapadas.** Hay que garantizar que todo el texto de una categoría caiga dentro de una sola ventana. El solape vuelve a mandar los mismos tokens para no cortar una unidad a la mitad.
3. **Modelos relativamente potentes.** Copiar cláusulas largas sin parafrasear, sin omitir y sin inventar exige un modelo que aguante bien la instrucción literal.

Eso produce cuatro problemas concretos:

- Se desaprovechan las ventanas de contexto grandes: la mayor parte del presupuesto se gasta en repetir el documento en la salida.
- Se reprocesan tokens. El solape vuelve a pagar el mismo texto en la llamada siguiente.
- El costo sube con la dificultad de la extracción, no solo con el tamaño del documento.
- Una ventana recortada empuja al modelo a cerrar la unidad como pueda. Ahí aparecen las alucinaciones: cláusulas parafraseadas, frases omitidas o texto que suena correcto y no está en la fuente.

## La salida con anclas

El sistema de anclas cambia el contrato. El LLM no busca el texto para reescribirlo. Mira dónde empieza la unidad y deja un ancla de inicio. Mira dónde termina y deja un ancla de fin. La salida se parece a esto:

```json
{
  "requirements": [
    {
      "start_anchor": "Breach means the acquisition, access, use, or disclosure of protected health information",
      "end_anchor": "the security or privacy of the protected health information.",
      "status": "complete"
    }
  ]
}
```

`start_anchor` y `end_anchor` son substrings literales del documento, cortos (unas doce a veinticinco palabras), no el párrafo. El motor localiza esas cadenas dentro del texto fuente y corta el span. El cuerpo de la unidad lo produce el código, no el modelo.

Con eso cambia el presupuesto:

- **Menos tokens de salida.** La respuesta son fronteras, no el documento otra vez.
- **Ventanas de entrada más grandes.** Al no reservar la salida para copiar el cuerpo, cabe más documento por llamada.
- **Sin solape para recuperar texto.** No hace falta reenviar el mismo pasaje para que el modelo lo copie entero. Si una unidad cruza el corte del chunk, el motor guarda una sola unidad pendiente y la cierra en el chunk siguiente, hacia adelante.
- **Modelos más ligeros.** La tarea es señalar fronteras verbatim, no redactar ni sostener un párrafo largo sin alterarlo.
- **El cuerpo no se alucina.** El modelo no escribe el contenido. Si el ancla no está en el documento, o no es única, el motor no inventa el párrafo: deja el segmento sin resolver y lo marca. El texto que sí sale es un corte del fuente, buscado con las anclas.



