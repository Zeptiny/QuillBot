import re

# Pre-compiled error patterns with responses
# Each tuple: (compiled_regex, response_template)
patterns = []

# Order matters: the first matching pattern wins. More specific patterns should
# appear before generic ones within each category.
_raw_patterns = {
    # === Plugin Errors ===
    r"Ambiguous plugin name `(\S+)' for files `(plugins\/[^\']+)' and `(plugins\/[^\']+)' in `plugins'":
    "Há múltiplos arquivos de plugins para o plugin '{0}': '{1}' e '{2}'.",

    r"Could not load 'plugins/([^']*)' in 'plugins'.*?Unknown/missing dependency plugins: \[([^\]]*)\]":
    "Não foi possível carregar o plugin `{0}` devido à dependência ausente `{1}`.",

    r"Error occurred while enabling ([^\s]+).*?([\w.]+Exception: .+)":
    "O plugin `{0}` encontrou um erro ao ser ativado: `{1}`.",

    r"Could not pass event (\S+) to (\S+)":
    "Erro ao processar o evento `{0}` no plugin `{1}`. Verifique se o plugin está atualizado e compatível.",

    # === JAR / Startup Errors ===
    r"Unable to access jarfile ([^'\n]*)":
    "Não foi possível acessar o arquivo jar `{0}` do servidor, verifique nos seus parâmetros de inicialização e as permissões.",

    r"Current Java is ([^\s]*) but we require at least ([^\s]*)":
    "O seu servidor pede a versão Java `{1}`, porém está configurado para iniciar com a versão `{0}`.",

    r"Unsupported Java detected \(([^)]+)\)\. Only up to Java (\d+) is supported":
    "Versão Java `{0}` não suportada. A versão máxima suportada é Java `{1}`.",

    r"Error: A JNI error has occurred":
    "Ocorreu um erro JNI. Isso geralmente indica incompatibilidade da versão Java com o servidor.",

    # === EULA ===
    r"You need to agree to the EULA in order to run the server":
    "Você precisa aceitar o EULA! Abra o arquivo `eula.txt` e altere `eula=false` para `eula=true`.",

    # === Memory / Performance ===
    r"java\.lang\.OutOfMemoryError":
    "O servidor ficou sem memória (RAM). Aumente a memória alocada com `-Xmx` nos parâmetros de inicialização ou otimize o servidor.",

    r"Can't keep up! Is the server overloaded\?.*?Running (\d+)ms.*?behind":
    "O servidor está sobrecarregado, ficando `{0}ms` atrás. Considere otimizar plugins, reduzir view-distance ou aumentar recursos.",

    r"Server thread/WARN.*?Can't keep up":
    "O servidor está com dificuldade de manter os ticks em dia. Verifique plugins pesados e configurações de desempenho.",

    # === Port / Network ===
    r"FAILED TO BIND TO PORT":
    "Falha ao vincular à porta. Verifique se outro processo já está usando a porta ou se ela está configurada corretamente.",

    r"Perhaps a server is already running on that port":
    "A porta já está em uso. Verifique se outra instância do servidor está rodando.",

    # === World / Data ===
    r"Failed to load chunk at \[(-?\d+), (-?\d+)\]":
    "Falha ao carregar o chunk em `[{0}, {1}]`. O chunk pode estar corrompido.",

    r"Region file .* is truncated":
    "Arquivo de região corrompido. Faça backup e considere deletar/regenerar a região afetada.",

    r"Session lock is no longer valid":
    "O lock de sessão do mundo não é mais válido. Verifique se outra instância está acessando o mesmo mundo.",

    # === Permissions ===
    r"(\S+) was denied the command: (.+)":
    "O jogador `{0}` não tem permissão para o comando `{1}`.",

    # === Version Mismatch ===
    r"Outdated server! I'm still on (.+)":
    "O servidor está desatualizado na versão `{0}`. Atualize o servidor ou use um plugin de compatibilidade como ViaVersion.",

    r"Outdated client! Please use (.+)":
    "O cliente do jogador está desatualizado. A versão necessária é `{0}`.",

    # === Crash ===
    r"This crash report has been saved to: (.+)":
    "Um crash report foi salvo em `{0}`. Analise o arquivo para mais detalhes.",

    r"---- Minecraft Crash Report ----":
    "O servidor crashou. Verifique o crash report para detalhes sobre a causa.",

    # === Generic ===
    "The received string length is longer than maximum allowed":
    "Você, ou o servidor, contém um nickname, scoreboard, tag, ou similar, com um tamanho maior do que o permitido.",

    r"Connection throttled! Please wait before reconnecting":
    "Conexão limitada. O jogador está tentando reconectar rápido demais.",
}

# Pre-compile all patterns at import time
for _pattern, _template in _raw_patterns.items():
    patterns.append((re.compile(_pattern), _template))