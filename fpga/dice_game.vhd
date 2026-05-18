library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- =============================================================================
-- UART TX Component
-- 8N1, 115200 baud at 100MHz (CLKS_PER_BIT = 868)
-- =============================================================================
entity uart_tx is
    Generic (
        CLKS_PER_BIT : INTEGER := 868
    );
    Port (
        clk      : in  STD_LOGIC;
        tx_start : in  STD_LOGIC;
        tx_byte  : in  STD_LOGIC_VECTOR(7 downto 0);
        tx_busy  : out STD_LOGIC;
        tx_pin   : out STD_LOGIC
    );
end uart_tx;

architecture Behavioral_uart of uart_tx is
    type t_uart_state is (IDLE, START_BIT, DATA_BITS, STOP_BIT);
    signal state     : t_uart_state := IDLE;
    signal clk_count : INTEGER range 0 to 868 := 0;
    signal bit_index : INTEGER range 0 to 7 := 0;
    signal tx_data   : STD_LOGIC_VECTOR(7 downto 0) := (others => '0');
begin
    process(clk)
    begin
        if rising_edge(clk) then
            case state is
                when IDLE =>
                    tx_pin    <= '1';
                    tx_busy   <= '0';
                    clk_count <= 0;
                    bit_index <= 0;
                    if tx_start = '1' then
                        tx_data <= tx_byte;
                        tx_busy <= '1';
                        state   <= START_BIT;
                    end if;

                when START_BIT =>
                    tx_pin <= '0';
                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        state     <= DATA_BITS;
                    end if;

                when DATA_BITS =>
                    tx_pin <= tx_data(bit_index);
                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        if bit_index < 7 then
                            bit_index <= bit_index + 1;
                        else
                            bit_index <= 0;
                            state     <= STOP_BIT;
                        end if;
                    end if;

                when STOP_BIT =>
                    tx_pin <= '1';
                    if clk_count < CLKS_PER_BIT - 1 then
                        clk_count <= clk_count + 1;
                    else
                        clk_count <= 0;
                        tx_busy   <= '0';
                        state     <= IDLE;
                    end if;
            end case;
        end if;
    end process;
end Behavioral_uart;


-- =============================================================================
-- DICE GAME TOP ENTITY
-- =============================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity dice_game is
    Port (
        clk : in  STD_LOGIC;
        rst : in  STD_LOGIC;
        btn : in  STD_LOGIC;
        sw  : in  STD_LOGIC_VECTOR(2 downto 0);
        seg : out STD_LOGIC_VECTOR(6 downto 0);
        an  : out STD_LOGIC_VECTOR(3 downto 0);
        tx  : out STD_LOGIC  -- UART TX to PC (pin D4 on Basys 3)
    );
end dice_game;

architecture Behavioral of dice_game is

    -- -------------------------------------------------------------------------
    -- UART component declaration
    -- -------------------------------------------------------------------------
    component uart_tx is
        Generic ( CLKS_PER_BIT : INTEGER := 868 );
        Port (
            clk      : in  STD_LOGIC;
            tx_start : in  STD_LOGIC;
            tx_byte  : in  STD_LOGIC_VECTOR(7 downto 0);
            tx_busy  : out STD_LOGIC;
            tx_pin   : out STD_LOGIC
        );
    end component;

    -- -------------------------------------------------------------------------
    -- LFSR
    -- Polynomial: x^16 + x^14 + x^13 + x^11 + 1
    -- -------------------------------------------------------------------------
    signal lfsr : STD_LOGIC_VECTOR(15 downto 0) := "1011010110101101";

    -- -------------------------------------------------------------------------
    -- Game signals
    -- -------------------------------------------------------------------------
    signal dice_p1, dice_p2 : INTEGER range 1 to 6 := 1;
    signal counter           : INTEGER := 0;
    constant MAX_COUNT       : INTEGER := 100000000; -- 1 second at 100MHz

    -- -------------------------------------------------------------------------
    -- Debounce / edge detection
    -- -------------------------------------------------------------------------
    signal btn_sync          : STD_LOGIC_VECTOR(1 downto 0) := (others => '0');
    signal btn_stable        : STD_LOGIC := '0';
    signal btn_debounced_reg : STD_LOGIC := '0';
    signal btn_pulse         : STD_LOGIC := '0';
    signal db_counter        : INTEGER range 0 to 1000000 := 0;

    -- -------------------------------------------------------------------------
    -- Display
    -- -------------------------------------------------------------------------
    signal refresh_counter : INTEGER := 0;
    signal digit_select    : INTEGER range 0 to 3 := 0;
    signal digit_value     : INTEGER range 0 to 9 := 0;

    -- -------------------------------------------------------------------------
    -- Adaptive bias
    -- -------------------------------------------------------------------------
    signal win_diff : INTEGER := 0;

    -- -------------------------------------------------------------------------
    -- UART / packet sender signals
    -- -------------------------------------------------------------------------
    signal uart_tx_start : STD_LOGIC := '0';
    signal uart_tx_byte  : STD_LOGIC_VECTOR(7 downto 0) := (others => '0');
    signal uart_tx_busy  : STD_LOGIC := '0';

    -- Latched values captured at the moment of each new roll
    signal dice_p1_latch : INTEGER range 1 to 6 := 1;
    signal dice_p2_latch : INTEGER range 1 to 6 := 1;
    signal mode_latch    : STD_LOGIC_VECTOR(2 downto 0) := (others => '0');

    -- One-cycle pulse that tells the packet FSM a new roll just happened
    signal send_trigger  : STD_LOGIC := '0';

    -- Packet sender FSM states
    -- Packet format: "P1:X,P2:Y,M:Z\n"  (14 ASCII bytes)
    type t_send_state is (
        WAIT_TRIGGER,
        SEND_P1_P, SEND_P1_1, SEND_P1_COLON, SEND_P1_VAL,
        SEND_COMMA1,
        SEND_P2_P, SEND_P2_2, SEND_P2_COLON, SEND_P2_VAL,
        SEND_COMMA2,
        SEND_M_CHAR, SEND_M_COLON, SEND_M_VAL,
        SEND_NEWLINE
    );
    signal send_state : t_send_state := WAIT_TRIGGER;

    -- -------------------------------------------------------------------------
    -- Helper functions
    -- -------------------------------------------------------------------------
    function to_7seg(val : INTEGER) return STD_LOGIC_VECTOR is
    begin
        case val is
            when 0      => return "1000000";
            when 1      => return "1111001";
            when 2      => return "0100100";
            when 3      => return "0110000";
            when 4      => return "0011001";
            when 5      => return "0010010";
            when 6      => return "0000010";
            when others => return "1111111";
        end case;
    end function;

    -- Converts integer 1-6 to its ASCII character ('1'=0x31 ... '6'=0x36)
    function int_to_ascii(val : INTEGER) return STD_LOGIC_VECTOR is
    begin
        return STD_LOGIC_VECTOR(to_unsigned(val + 48, 8));
    end function;

    -- Converts 3-bit mode vector to ASCII digit ('0'=0x30 ... '4'=0x34)
    function mode_to_ascii(m : STD_LOGIC_VECTOR(2 downto 0)) return STD_LOGIC_VECTOR is
    begin
        return STD_LOGIC_VECTOR(to_unsigned(to_integer(unsigned(m)) + 48, 8));
    end function;

begin

    -- =========================================================================
    -- UART TX instantiation
    -- =========================================================================
    UART_INST : uart_tx
        generic map ( CLKS_PER_BIT => 868 )
        port map (
            clk      => clk,
            tx_start => uart_tx_start,
            tx_byte  => uart_tx_byte,
            tx_busy  => uart_tx_busy,
            tx_pin   => tx
        );

    -- =========================================================================
    -- 1. LFSR RANDOM GENERATOR
    -- =========================================================================
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                lfsr <= "1011010110101101";
            else
                lfsr <= lfsr(14 downto 0) & (lfsr(15) xor lfsr(13) xor lfsr(12) xor lfsr(10));
            end if;
        end if;
    end process;

    -- =========================================================================
    -- 2. BUTTON DEBOUNCE
    -- =========================================================================
    process(clk)
    begin
        if rising_edge(clk) then
            btn_sync <= btn_sync(0) & btn;

            if btn_sync(1) /= btn_stable then
                db_counter <= 0;
                btn_stable <= btn_sync(1);
            elsif db_counter < 1000000 then
                db_counter <= db_counter + 1;
            else
                btn_debounced_reg <= btn_stable;
            end if;

            btn_pulse <= btn_stable and (not btn_debounced_reg);
        end if;
    end process;

    -- =========================================================================
    -- 3. GAME LOGIC
    --    send_trigger is pulsed for exactly one clock cycle on every new roll.
    -- =========================================================================
    process(clk)
        variable r1, r2          : INTEGER;
        variable bias_threshold  : INTEGER;
    begin
        if rising_edge(clk) then
            send_trigger <= '0';  -- default: no trigger this cycle

            if rst = '1' then
                dice_p1  <= 1;
                dice_p2  <= 1;
                counter  <= 0;
                win_diff <= 0;
            else
                r1 := (to_integer(unsigned(lfsr(7 downto 0)))  mod 6) + 1;
                r2 := (to_integer(unsigned(lfsr(15 downto 8))) mod 6) + 1;

                case sw is

                    -- MANUAL MODE
                    when "000" =>
                        if btn_pulse = '1' then
                            dice_p1      <= r1;
                            dice_p2      <= r2;
                            send_trigger <= '1';
                        end if;

                    -- AUTO MODE (1 roll per second)
                    when "001" =>
                        if counter >= MAX_COUNT then
                            counter      <= 0;
                            dice_p1      <= r1;
                            dice_p2      <= r2;
                            send_trigger <= '1';
                        else
                            counter <= counter + 1;
                        end if;

                    -- BIAS P1 (70% chance of rolling a 5)
                    when "010" =>
                        if counter >= MAX_COUNT then
                            counter <= 0;
                            if (to_integer(unsigned(lfsr(7 downto 0))) mod 10) < 7 then
                                dice_p1 <= 5;
                            else
                                dice_p1 <= r1;
                            end if;
                            dice_p2      <= r2;
                            send_trigger <= '1';
                        else
                            counter <= counter + 1;
                        end if;

                    -- CHAOS MODE (5x speed, still fair)
                    when "011" =>
                        if counter >= MAX_COUNT / 5 then
                            counter      <= 0;
                            dice_p1      <= r1;
                            dice_p2      <= r2;
                            send_trigger <= '1';
                        else
                            counter <= counter + 1;
                        end if;

                    -- ADAPTIVE BIAS
                    when "100" =>
                        if counter >= MAX_COUNT then
                            counter <= 0;

                            if dice_p1 > dice_p2 then
                                win_diff <= win_diff + 1;
                            elsif dice_p2 > dice_p1 then
                                win_diff <= win_diff - 1;
                            end if;

                            if win_diff > 5 then
                                bias_threshold := 40;
                            elsif win_diff < -5 then
                                bias_threshold := 70;
                            else
                                bias_threshold := 55;
                            end if;

                            if (to_integer(unsigned(lfsr(15 downto 8))) mod 100) < bias_threshold then
                                dice_p1 <= (r1 mod 3) + 4;
                            else
                                dice_p1 <= r1;
                            end if;
                            dice_p2      <= r2;
                            send_trigger <= '1';
                        else
                            counter <= counter + 1;
                        end if;

                    when others => null;
                end case;
            end if;
        end if;
    end process;

    -- =========================================================================
    -- 4. DISPLAY REFRESH COUNTER
    -- =========================================================================
    process(clk)
    begin
        if rising_edge(clk) then
            if refresh_counter >= 50000 then
                refresh_counter <= 0;
                if digit_select = 3 then
                    digit_select <= 0;
                else
                    digit_select <= digit_select + 1;
                end if;
            else
                refresh_counter <= refresh_counter + 1;
            end if;
        end if;
    end process;

    -- =========================================================================
    -- 5. DIGIT SELECT LOGIC
    -- =========================================================================
    process(digit_select, dice_p1, dice_p2)
    begin
        case digit_select is
            when 0 =>
                an          <= "1110";
                digit_value <= dice_p2;
            when 1 =>
                an          <= "1101";
                digit_value <= 0;
            when 2 =>
                an          <= "1011";
                digit_value <= 0;
            when others =>
                an          <= "0111";
                digit_value <= dice_p1;
        end case;
    end process;

    seg <= to_7seg(digit_value);

    -- =========================================================================
    -- 6. UART PACKET SENDER FSM
    --    Triggered by send_trigger pulse from game logic.
    --    Waits for uart_tx_busy='0' before sending each byte.
    --    Packet: "P1:X,P2:Y,M:Z\n"
    --
    --    ASCII reference used below:
    --      'P'=0x50  '1'=0x31  '2'=0x32  ':'=0x3A
    --      ','=0x2C  'M'=0x4D  '\n'=0x0A
    -- =========================================================================
    process(clk)
    begin
        if rising_edge(clk) then
            uart_tx_start <= '0';  -- default: no start pulse

            -- Latch dice values and mode the moment a new roll fires.
            -- This keeps the packet consistent even if dice_p1/p2 change
            -- before the multi-byte transmission finishes.
            if send_trigger = '1' then
                dice_p1_latch <= dice_p1;
                dice_p2_latch <= dice_p2;
                mode_latch    <= sw;
                send_state    <= SEND_P1_P;
            end if;

            case send_state is

                when WAIT_TRIGGER =>
                    null;

                -- --- "P" ---
                when SEND_P1_P =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"50";  -- 'P'
                        uart_tx_start <= '1';
                        send_state    <= SEND_P1_1;
                    end if;

                -- --- "1" ---
                when SEND_P1_1 =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"31";  -- '1'
                        uart_tx_start <= '1';
                        send_state    <= SEND_P1_COLON;
                    end if;

                -- --- ":" ---
                when SEND_P1_COLON =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"3A";  -- ':'
                        uart_tx_start <= '1';
                        send_state    <= SEND_P1_VAL;
                    end if;

                -- --- dice_p1 digit ---
                when SEND_P1_VAL =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= int_to_ascii(dice_p1_latch);
                        uart_tx_start <= '1';
                        send_state    <= SEND_COMMA1;
                    end if;

                -- --- "," ---
                when SEND_COMMA1 =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"2C";  -- ','
                        uart_tx_start <= '1';
                        send_state    <= SEND_P2_P;
                    end if;

                -- --- "P" ---
                when SEND_P2_P =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"50";  -- 'P'
                        uart_tx_start <= '1';
                        send_state    <= SEND_P2_2;
                    end if;

                -- --- "2" ---
                when SEND_P2_2 =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"32";  -- '2'
                        uart_tx_start <= '1';
                        send_state    <= SEND_P2_COLON;
                    end if;

                -- --- ":" ---
                when SEND_P2_COLON =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"3A";  -- ':'
                        uart_tx_start <= '1';
                        send_state    <= SEND_P2_VAL;
                    end if;

                -- --- dice_p2 digit ---
                when SEND_P2_VAL =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= int_to_ascii(dice_p2_latch);
                        uart_tx_start <= '1';
                        send_state    <= SEND_COMMA2;
                    end if;

                -- --- "," ---
                when SEND_COMMA2 =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"2C";  -- ','
                        uart_tx_start <= '1';
                        send_state    <= SEND_M_CHAR;
                    end if;

                -- --- "M" ---
                when SEND_M_CHAR =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"4D";  -- 'M'
                        uart_tx_start <= '1';
                        send_state    <= SEND_M_COLON;
                    end if;

                -- --- ":" ---
                when SEND_M_COLON =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"3A";  -- ':'
                        uart_tx_start <= '1';
                        send_state    <= SEND_M_VAL;
                    end if;

                -- --- mode digit ---
                when SEND_M_VAL =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= mode_to_ascii(mode_latch);
                        uart_tx_start <= '1';
                        send_state    <= SEND_NEWLINE;
                    end if;

                -- --- "\n" (0x0A) ---
                when SEND_NEWLINE =>
                    if uart_tx_busy = '0' then
                        uart_tx_byte  <= x"0A";  -- newline
                        uart_tx_start <= '1';
                        send_state    <= WAIT_TRIGGER;
                    end if;

            end case;
        end if;
    end process;

end Behavioral;