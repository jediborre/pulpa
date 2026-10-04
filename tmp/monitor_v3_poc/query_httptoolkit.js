/**
 * Script auxiliar: query_httptoolkit.js
 * Propósito: Consultar el puente de control IPC (Named Pipe //./pipe/httptoolkit-ctl)
 * de HTTP Toolkit para extraer las peticiones HTTP/HTTPS interceptadas, filtrando
 * por sofascore. Permite obtener los eventos y extraer las cabeceras exactas.
 */

const http = require('http');

function postJson(path, body) {
    return new Promise((resolve, reject) => {
        const payload = JSON.stringify(body);
        const req = http.request({
            socketPath: '//./pipe/httptoolkit-ctl',
            path: path,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload)
            }
        }, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(data));
                } catch (e) {
                    resolve(data);
                }
            });
        });
        req.on('error', reject);
        req.write(payload);
        req.end();
    });
}

async function main() {
    console.log('Consultando eventos capturados en HTTP Toolkit...');
    
    // 1. Listar eventos generales o de sofascore
    const listResult = await postJson('/api/execute', {
        name: 'events.list',
        source: 'mcp',
        args: {
            filter: 'hostname*=sofascore',
            limit: 20
        }
    });

    console.log('Resultado eventos sofascore:', JSON.stringify(listResult, null, 2));

    const events = listResult.data?.events || listResult.events || [];
    console.log(`Se encontraron ${events.length} eventos de SofaScore.`);

    if (events.length === 0) {
        console.log('No se encontraron eventos de SofaScore.');
        return;
    }

    for (const ev of events.slice(0, 5)) {
        console.log(`\n========================================`);
        console.log(`Evento: ${ev.id}`);
        console.log(`Method: ${ev.method} | Status: ${ev.status}`);
        console.log(`URL: ${ev.url}`);

        const outlineResult = await postJson('/api/execute', {
            name: 'events.get-outline',
            source: 'mcp',
            args: { id: ev.id }
        });

        const outline = outlineResult.data || outlineResult;
        console.log('--- HEADERS DE PETICIÓN ---');
        console.log(JSON.stringify(outline.request?.headers, null, 2));

        if (ev.url.includes('token/init')) {
            const bodyResult = await postJson('/api/execute', {
                name: 'events.get-response-body',
                source: 'mcp',
                args: { id: ev.id }
            });
            console.log('--- RESPONSE BODY TOKEN/INIT ---');
            console.log(JSON.stringify(bodyResult.data || bodyResult, null, 2));
        }
    }
}

main().catch(err => console.error('Error fatal:', err));
